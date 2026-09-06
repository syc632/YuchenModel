import re
from pathlib import Path
from dataclasses import dataclass
from transformers import AutoTokenizer, AutoModel
from torch.utils.data import DataLoader
import torch
import torch.distributed as dist
import torch.nn.functional as f
from transformers import AutoTokenizer
from contextlib import nullcontext
from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler,Dataset
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from model.model import YuchenModelCausalLLM,Config
from train.train_util import *


@dataclass
class PPOConfig:
    # 数据与奖励模型
    data_file: str = "data/rlaif.jsonl"
    reward_model_path: str = "reward_model.pth"
    max_length: int = 1024
    max_new_tokens: int = 1000
    is_reasoning: bool = False



    # PPO 目标
    clip_epsilon: float = 0.5
    vf_coef: float = 0.5
    kl_coef: float = 0.5
    update_old_actor: int = 100



    # 优化与训练批次
    epochs: int = 1
    batch_size: int = 8
    accumulation_steps: int = 10
    lr: float = 1e-4
    grad_clip: float = 1.0
    total_optimizer_steps: int = 10000
    num_worker: int = 2
    dtype: str = "bfloat16"



    # 模型与运行环境
    hidden_size: int = 512
    use_moe: bool = True
    use_compile: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"



    # 断点、保存与实验记录
    save_dir: str = ""
    save_weight: str = ""
    use_resume: bool = True
    save_interval: int = 1000
    wandb_proj: str = "YuchenModel"

class PPODataset(Dataset):
    def __init__(self,tokenizer,cfg:PPOConfig):
        #分词器
        self.tokenizer = tokenizer
        self.max_length = cfg.max_length
        #使用HuggingFace的load_data来加载数据集,
        self.samples = load_dataset("json",data_files=cfg.data_file,split="train")


        #获取特殊token的ID,这里尝试获取BOS的input_ids
        self.bos_id = tokenizer(f"{tokenizer.bos_token}assistant",add_special_token=False).input_ids
        self.eos_id = tokenizer(f"{tokenizer.eos_token}",add_special_token = False).input_ids


    def __len__(self):
        return len(self.samples)

    def create_chat_prompt(self,conversation):
        #用于存放标准的对话格式,如:[{"role":"user","content":"xxx"},....]
        message = []
        #用于存放最终的回复内容
        answer = ""

        #遍历原始对话列表
        for i,turn in enumerate(conversation):
            #根据索引判断角色,默认偶数是user(0,2,4,..),奇数是assistant
            role = "user" if i % 2 == 0 else "assistant"
            message.append({"role":role,"content":turn["content"]})

            #不断覆盖answer,当循环结束的时候,保存的就是对话列表的最后一句话
            answer = turn["content"]


            #使用tokenizer的chat_ template功能将历史对话格式化为单个字符串
        return self.tokenizer.apply_chat_template(
                message[:-1], #除了最后一句话以外的所有对话
                tokenize = False,    #false表示返回字符串,而不是token id
                add_generation_prompt=True  #True表示在字符串末尾自动加上让模型开始回答的提示(如:"assistant\n"
            ),answer

    #根据索引获取单条数据,DataLoader拉取数据的核心方法
    def __getitem__(self, item):
        #取出指定索引处的原始数据字典
        sample = self.samples[item]


        #提取出样本中的'conversation'字段,用create_chat_prompt进行格式化
        prompt,answer = self.create_chat_prompt(conversation=sample["conversations"])


        return{"prompt":prompt,"answer":answer}



#critic model:评价当前的state value
class CriticModel(YuchenModelCausalLLM):
    def __init__(self,param):
        super().__init__(param)
        #将原有的输出头替换为一个线性层,该线性层将隐藏状态映射为一个标量
        self.value_head = nn.Linear(param.d_model, 1)

    def forward(self,input_ids = None,attention_mask = None,**kwargs):
        """
        因为送入网络的是一堆token ID,所以需要使用Embedding
        """
        if attention_mask is None:
            if self.config.pad_token_id is None:
                padding_mask = torch.ones_like(input_ids,dtype=torch.bool)
            else:
                padding_mask = (input_ids != self.config.pad_token_id)

        input_token = self.embd(input_ids)
        input_token = input_token
        hidden_state,_,_ = self.model(input_token,padding_mask = padding_mask)

        #B L D --> B L 1 --> B L
        hidden_state = self.value_head(hidden_state).squeeze(-1)

        return hidden_state

def calculate_rewards(prompts,responses,reward_model,reward_tokenizer,device):
    """
    整合所有奖励函数的总奖励
    """
    #针对推理模型的特殊奖励计算逻辑
    def reasoning_model_reward(rewards):


        #1.格式奖励,查看输出是否带有<think>思考过程</think> <answer>回答过程</answer>
        # ^表示字符串开头,前面不能有任何内容
        # . 任意字符
        # *重复0或多次
        #? 尽可能少的匹配
        # $表示字符串结尾,说明</answer>结尾不能有任何东西
        pattern = r"^<think>\n.*?\n</think>\n<answer>\n.*?</answer>$"
        pattern2 = r"^<think>\n.*?\n</think>\n\n<answer>.*?</answer>$"   #接受空行

        #re:regular expression
        #re.match:从字符串开头开始匹配正则模式
        #re.S:re.DOTALL的缩写,让正则中的"."(点号)能匹配所有字符(包括\n)
        match_pattern = [re.match(pattern,response,re.S) for response in responses]
        match_pattern2 = [re.match(pattern2,response,re.S) for response in responses]

        format_reward = []
        for match_pattern,match_pattern2 in zip(match_pattern,match_pattern2):
            if match_pattern:
                format_reward.append(0.5)
            elif match_pattern2:
                format_reward.append(0.5)
            else:
                format_reward.append(0.0)

        rewards += torch.tensor(format_reward,device=device)





        #2.token奖励:为了防止信号稀疏导致的训练困难,为出现的特定的token给部分奖励
        def mark_num(text):
            """
            统计一下response总出现多少次特殊token,并给予对应的奖励
            """
            reward = 0
            if text.count("<think>") == 1: reward += 0.25
            if text.count("</think>") == 1: reward += 0.25
            if text.count("<answer>") == 1: reward += 0.25
            if text.count("</answer>") == 1: reward += 0.25
            return reward
        #计算response中获得的所有奖励
        mark_reward = [mark_num(response) for response in responses]
        rewards += torch.tensor(mark_reward,device=device)
        return rewards



    #初始化每个reward为0,维度为[B]
    reward = torch.zeros(len(responses),device=device)
    #如果是推理模型,先加入基于规则的奖励
    if cfg.is_reasoning:
        reward = reasoning_model_reward(reward)


    #3.使用奖励模型对语义和内容质量进行打分
    with torch.no_grad():
        reward_model_score = []
        for prompt,response in zip(prompts,responses):
            #(system|user|assistant):匹配并捕获角色
            #\s+"匹配一个或多个空白符号,如空格换行
            #匹配格式:
            #     <\|im_start\|>user
            #           你好,介绍一下PPO
            #     <\|im_start\|>user
            pattern = r"<\|im_start\|>(system|user|assistant)\s+(.*?)<\|im_end\|>"
            #re.findall:查找所有匹配项的函数,在prompt中寻找符合pattern的内容,并返回一个列表
            matches = re.findall(pattern, prompt,re.DOTALL)
            messages = [{"role": role, "content": content} for role, content in matches]

            #将模型当前生成的response拼回去,构造完整的对话记录
            tmp_chat = messages + [{"role": "assistant", "content": response}]
            #调用奖励模型的API进行打分,这里是对整个句子进行打分
            score = reward_model.get_score(reward_tokenizer,tmp_chat)

            #限制得分在(-3.0,3.0)之间,防止异常大/小的数值导致训练崩溃
            scale = 3.0
            #min(score,scale)小于scale
            #max(...,-scale)大于-scale
            score = max(min(score,scale),-scale)

            #如果是推理模型,不仅要对整体打分,还要对答案进行单独打分
            #整体得分:鼓励模型进行合理的推理,遵守输出格式
            #答案得分:最终保证答案本身正确,清洗
            if cfg.is_reasoning:
                answer_match = re.match("<answer>",response,re.DOTALL)
                if answer_match:
                    #对于 <answer>答案 </answer>
                    #group(1)为答案
                    answer_content = answer_match.group(1).strip()
                    #针对纯答案计算reward
                    tmp_chat = messages + [{"role":"assistant","content":answer_content}]
                    score = reward_model.get_score(reward_tokenizer,tmp_chat)
                    answer_score = max(min(score,scale),-scale)
                    #加权合并
                    score = 0.4*score + 0.6*answer_score

            reward_model_score.append(score)

        #转换为张量并累加到总奖励上
        reward_model_score = torch.tensor(reward_model_score,device=device)
        reward += reward_model_score

    return reward


def ppo_train_one_epoch(cfg:PPOConfig,loader,epoch,iter,old_actor_model,critic_model,actor_model,reward_model,reward_tokenizer,
                        ref_model,actor_scheduler,critic_scheduler,tokenizer,start_step=0,wandb=None):
    """

    :param cfg:
    :param loader:
    :param iter:
    :param old_actor_model: 更新前的旧actor_model,用他的旧的策略概率,去计算PPO的重要性比率
    :param reward_model: 奖励模型,评价actor_model生成的回答
    :param reward_tokenizer: 奖励模型的分词器
    :param ref_model: 参考模型,一般是基座模型,和actor_model比较计算KL散度,避免基座模型偏离太远
    :param actor_scheduler: 优化器的学习率调度器,负责调整学习率
    :param critic_scheduler:
    :param start_step:
    :param wandb:
    :return:

    流程:
    Actor模型生成回答,Reward Model打分 ->critic估计价值 -> 根据PPO损失更新Actor和critic

    每个batch的流程:
    Actor生成Response -> Reward_model得到 reward -> Critic_model估计Value
    计算advantage Actor/Old Actor/Reference计算概率 -> 计算PPO Loss ->反向传播更新参数


    """

    for step,batch in enumerate(loader,start_step+1):


        #准备工作,把文本转为token ID
        prompts = batch["prompt"]  #获取当前批次的prompt列表
        enc = tokenizer(prompts,return_tensors = "pt",padding=True,truncation=True,
                        #使用左侧填充,使得prompt的最后一个位置是有效token
                        max_length = cfg.max_length,padding_side = "left").to(cfg.device)
        #enc.input_ids:[B,T]
        #padding后的统一长度
        prompt_length = enc.input_ids.shape[1]





        #1.Rollout采样,当前的Actor Model根据prompt进行自由回答
        #这个时候只是获取数据,不需要去计算梯度
        with torch.no_grad():
            #如果使用分布式训练,原来的actor_mdoel被DistributedDataParallel包裹了,需要 .module才能调用generate方法
            model_for_gen = actor_model.module if isinstance(actor_model,DistributedDataParallel) else actor_model
            #actor_model根据prompt生成response
            #这里的gen_out是完整的序列:prompt + actor生成的response
            gen_out = actor_model.generate(input_ids = enc.input_ids,attention_mask = enc.attention_mask,max_new_tokens = cfg.max_new_tokens,
                                           do_sample = True,temperature = 0.9,pad_token_id = tokenizer.pad_token_id,eos_token_id = tokenizer.eos_token_id)

        #将生成的token ID解码回纯文本(取出prompt部分)
        response_text = [tokenizer.decode(gen_out[i,prompt_length:]) for i in range(len(prompts))]





        #2.奖励与优势计算
        #第1步生成的回答给Reward Model进行打分
        #让critic model预测一下这个回答能得多少分
        #获取该batch生成的最终奖励 [B]
        reward = calculate_rewards(prompts,response_text,reward_model,reward_tokenizer,device=cfg.device)

        #生成full_mask来区分实际token和padding
        full_mask  = (gen_out != tokenizer.pad_token_id).long()

        #critic模型计算整个序列每个位置的state value
        #critic需要看到完整的序列才能判断模型生成的回答的每个状态的state value是多少
        value_seq = critic_model(input_ids = gen_out,attention_mask = full_mask)

        #critic_model输出的是每个位置的state_value,但是每条样本最终只有一个奖励,因此需要从每个序列的多个state value中
        #选出一个代表完整回答结束后状态的的value,这里选用的是最后一个有效位置的state value
        last_indices = (full_mask*torch.arange(full_mask.size(1),device=cfg.device)).argmax(dim=1) #B L,dim=1沿序列做
        #提取最后一个value作为整个回复的预测价值
        value = value_seq[torch.arange(gen_out.size(0)),last_indices]

        #计算优势函数 :A =  Reward - Baseline(V) -> [B]
        #.detach() 保证反向传播不会通过value影响此处计算
        advantages  = reward - value.detach()






        #3.Actor_model进行前向传播计算当前策略的概率和梯度
        #反思和计算梯度,把刚刚生成好的回答重新喂给actor_model,计算模型刚才刚才回答的对数概率
        #为什么不在前向传播的时候计算概率?
            #1.显存会报OOM
            #这一步需要用到掩码,因为prompt不是模型需要去修改的地方,模型只需要去修改自己response的概率即可
            #如果我们在采样的时候计算对数概率,那么Pytorch就必须在显存中保存完整的计算图,而大模型的生成是串行的,这导致很多前传的激活值必须堆在显卡中
            #对于一个十几亿的参数,即使生成100个token,显存也会报OOM
            #2.批处理效率高
            #在generate阶段,我们只需要生成文本,不需要计算梯度("纯推理")速度块,显存占用少,等待句子生成完了之后,我们得到了完整的gen_out
            #此时把所有的句子作为一个完整的序列一次性的全部送给模型,Transformer 架构的优势就在于并行计算，这样只需一次前向传播就能得出所有 1000 个
            # token 的对数概率并构建好用于反向传播的计算图，效率远高于在生成时循环算 1000 次
        with autocast_ctx:
            #将完整的序列输入到actor_model,获得完整的logits
            res = actor_model(input_ids = gen_out,attention_mask = full_mask)
            logits = res.logits #B P+R-1
            #如果是MoE模型,计算一下辅助损失
            aux_loss = res.aux_loss if cfg.use_moe else 0.0

        #label表示这条trajectory实际执行的action
        #Label为实际采样的token ID,使用这个token ID去索引实际action的概率
        #因为在采样概率的时候并没有保存计算图,因此不能用于反向传播,并且generate生成的是序列的token ID,并没有保存每个位置的logits
        label = gen_out[:,1:].clone() #错一位当作label, B P+R-1
        #logp_tokens
        #logits B P+R-1 V
        #label B +R-1 因此需要先升一维进行索引,把值取出来之后再降维即可
        logp_tokens = f.log_softmax(logits[:,:-1],dim=-1).gather(2,label.unsqueeze(-1)).squeeze(-1)


        #构造掩码,仅生成Response部分的logp计算,忽略Prompt和Padding部分
        #P+R-1
        seq_len = gen_out.size(1) - 1
        resp_mask = torch.arange(seq_len,device = cfg.device).unsqueeze(0) >= prompt_length -1
        #rq表示相等,~取反表示不相等
        final_mask = resp_mask & (~label.eq(tokenizer.pad_token_id))

        #当前策略下整个response的总对数概率分布
        actor_logp = (logp_tokens*final_mask).sum(dim=1)






        #4.old_model和ref_model进行前向传播
        #两个参照物:
        #   old_actor:用来看看模型更新前,生成这段话的概率是多少,用于计算重要性采样
        #   ref_model:用来防止模型为了拿高分而偏离基座模型太远,(比如无限重复某个高分词语),要求当前的策略不能偏离初始的策略太远
        with torch.no_grad():
            #旧策略模型(用于PPO重要性比率)
            old_logits = old_actor_model(input_ids = gen_out,attention_mask = full_mask).logits
            #gether使用label中的token ID去索引每个位置上实际action的对数概率
            old_logp_tokens = f.log_softmax(old_logits[:,:-1],dim=-1).gather(2,label.unsqueeze(-1)).squeeze(-1)
            old_logp = (old_logp_tokens*final_mask).sum(dim=-1)


            #参考模型(基座模型,用于防止策略过度偏移的KL,乘法计算)
            ref_logits = ref_model(input_ids = gen_out,attention_mask = full_mask).logits
            ref_logp_tokens = f.log_softmax(ref_logits[:,:-1],dim=-1).gather(2,label.unsqueeze(-1)).squeeze(-1)
            ref_logp = (ref_logp_tokens*final_mask).sum(dim=-1)




        #5.PPO计算损失
        # ratio:当前概率/旧概率
        #   surr1/surr2:(Clip操作):如果Advantage是正的,我们想要提高这个action概率,clamp限制这个动作更新幅度最多是1+σ
        #   critic_loss:让critic预测的分数尽可能的接近reward预测的分数
        #KL散度:actor_model和old_actor_model之间的分布差异(监控指标)
        #.mean(),再batch维度上取平均,即对所有的trajectory的平均对数概率
        kl = (actor_logp - old_logp).mean()

        #kl散度:actor_model与reference_model之间的分布差异,直接作为惩罚加入到Loss
        kl_ref = (actor_logp - ref_logp).mean()


        #计算重要性采样比率
        ratio = torch.exp(actor_logp - old_logp)


        #PPO的代理目标,用于计算Actor的策略损失函数
        #不剪裁的目标,直接根据当前策略相对于旧策略的变化来更新概率
        surr1 = ratio*advantages
        #裁剪项,将ratio限制[1- epsilon , 1 + epsilon]之间
        #限制策略概率最多变化一定幅度,防止一次更新过大
        surr2 = torch.clamp(ratio,1.0-cfg.clip_epsilon,1.0+cfg.clip_epsilon)*advantages
        #ActorLoss负的最小话(即最大化目标函数)
        policy_loss = -torch.minimum(surr1,surr2).mean()


        #CriticLoss :预测的价值Value 与实际Reward 之间的均方误差
        #reward和value本质上都是一个回归问题
        value_loss = f.mse_loss(reward,value)


        #总损失  =  策略损失 + 价值损失 + KL散度惩罚 + MoE损失
        #除以梯度累积步以做平均
        loss = (policy_loss + cfg.vf_coef*value_loss + cfg.kl_coef*kl_ref +aux_loss)/cfg.accumulation_steps
        loss.backward()





        #6.参数更新
        #套路:梯度剪裁 ---> 优化器修改参数 ---> 调度器调整学习率 ---> 情况梯度开始下一轮
        if step%cfg.accumulation_steps == 0:
            #梯度剪裁,防止爆炸
            torch.nn.utils.clip_grad_norm_(actor_model.parameters(),cfg.grad_clip)
            torch.nn.utils.clip_grad_norm_(critic_model.parameters(),cfg.grad_clip)
            actor_optimizer.step()
            critic_optimizer.step()
            critic_scheduler.step()
            #清空梯度
            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()





        #7.日志打印
        if is_main_process():
            #计算生成的平均模型
            #input_ids [B,P]
            response_ids = gen_out[:,enc.input_ids.shape[1]:]
            is_eos = (response_ids == tokenizer.eos_token_id)
            eos_indices = torch.argmax(is_eos.int(),dim=1)
            has_eos = eos_indices.any(dim=1)
            #torch.where(condition,A,B):条件成立,从A中取;条件不成立,从B中取
            #torch.where(condition)返回符合条件的元素的位置索引
            #如果有eos_id + 1:某条response实际生成到eos为止的长度
            avg_length = torch.where(has_eos,eos_indices+1,torch.tensor(response_ids.shape[1],device=is_eos.device))

            #取出各项指标用于打印日志
            actor_loss_val = policy_loss.item()
            critic_loss_val = value_loss.item()
            current_aux_loss = aux_loss.item()
            reward_val = reward.mean().item()
            kl_val = kl.item()
            kl_ref_val = kl_ref.item()
            avg_len_val = avg_length.item()
            actor_lr = actor_optimizer.param_group[0]['lr']
            critic_lr = critic_optimizer.param_group[0]['lr']


            #上传Wandb
            if wandb is not None:
                wandb.log({
                    "actor_loss":actor_loss_val,"critic_loss":critic_loss_val,
                    "aux_loss":current_aux_loss,"reward":reward_val,"kl":kl_val,
                    "kl_ref":kl_ref_val,"avg_len":avg_len_val,"actor_lr":actor_lr
                })


            #打印到控制台
            Logger(f"Epoch[{epoch+1}/{cfg.epochs}]({step}/{iter}"
                   f"Actor_loss:{actor_loss_val:.4f},CriticLoss:{critic_loss_val:.4f},Aux_loss:{current_aux_loss:.4f}"
                   f"Reward:{reward_val:.4f},KL:{kl_val:.4f},KL_ref:{kl_ref_val:.4f},AvgResponseLen:{avg_len_val:.2f}"
                   f"Actor LR:{actor_lr:.8f},Critic Lr:{critic_lr:.8f}")





        #8.模型状态同步
        if step%cfg.update_old_actor == 0:
            raw_actor = actor_model.module if isinstance(actor_model,DistributedDataParallel) else actor_model
            raw_actor = getattr(raw_actor,"orig_mod",raw_actor)
            state_dict = raw_actor.state_dict()
            #拷贝一份到CPU,并加载到old_actor,防止直接关联显存中的计算图
            #数据流程 : Actor(GPU)的参数 --.detach().cpu()--> CPU上的一份权重快照 -- load_state_dict() --> old_actor_model的参数 --to(device) -->继续放回GPU推理
            old_actor_model.load_state_dict({k:v.detach().cpu() for k,v in state_dict.item()})
            old_actor_model.to(device = cfg.device)





        #9.模型保存
        #定期保存,或者达到当前Epoch或者最后一个iter保存
        if (step%cfg.save_interval == 0 or step == iter-1) and is_main_process():
            actor_model.eval()
            moe_suffix = '_moe' if cfg.use_moe else ''
            ckp = f"{cfg.save_dir}/{cfg.hidden_size}_{cfg.hidden_size}{moe_suffix}.pth"
            #提取原模型(解除 DDP 包装)
            raw_actor = actor_model.module if isinstance(actor_model,DistributedDataParallel) else actor_model
            #如果raw_actor有orig_mod属性,就取出它,否则保持raw_actor不变
            raw_actor = getattr(raw_actor,'_orig_mod',raw_actor)
            actor_state = raw_actor.state_dict()
            #转为半精度存到硬盘,节省空间
            torch.save({k:v.half().cpu() for k,v in actor_state.items()},ckp)

            #使用工具函数保存完整的训练状态检查点(支持断点续训)
            lm_check_point(lm_config=PPOConfig,weight=cfg.save_weight,model=actor_model,optimizer=actor_optimizer,epoch=epoch,
                           step = step,wandb = wandb,save_dir = ".../checkpoint",scheduler = actor_scheduler,critic_model = critic_model,
                           critic_scheduler = critic_scheduler)

            #保存完返回训练
            actor_model.train()
            #释放内存
            del actor_state


        #10.清理内存
        #RLHF一个Epoch中既有推理又有前反传,如果不把张量从显存中提出去,就会报OOM
        del enc,gen_out,response_text,reward,full_mask,value_seq,value,advantages
        del logits,label,logp_tokens,final_mask,actor_logp,old_logits,old_logp,ref_logits,ref_logp
        del kl,kl_ref,ratio,surr1,surr2,policy_loss,value_loss,loss





if __name__ == "__main__":
    _root = Path(r"D:\Kimi")
    tokenizer = AutoTokenizer.from_pretrained(str(_root / "BPEmodel"))
    #
    # rlaif_path = _root / "data" / "rlaif.jsonl"
    # if not rlaif_path.is_file():
    #     print("未找到 rlaif.jsonl:", rlaif_path)
    #     raise SystemExit(1)
    #
    cfg = PPOConfig()
    # cfg.data_file = str(rlaif_path)
    # ds = PPODataset(tokenizer, cfg)
    # if len(ds) == 0:
    #     print("RLAIF 数据集为空")
    #     raise SystemExit(1)
    #
    # for i in range(min(2, len(ds))):
    #     print(ds[i])


    #1.初始化环境和种子
    # 初始化分布式通信后端，获得当前设备的 local_rank
    local_rank = init_distributed_mode()
    if dist.is_initialized(): cfg.device = f"cuda:{local_rank}"
    # 设置随机种子，保证所有进程初始化一致
    set_seed(42 + (dist.get_rank() if dist.is_initialized() else 0))




    #2.配置目录,模型参数,检查点
    os.makedirs(cfg.save_dir,exist_ok=True)
    model_config = Config()
    #检查是否有之前保存的checkpoint用于断点续训
    ckp_data = lm_check_point(lm_config=model_config,weight=cfg.save_weight,save_dir = "../checkpoint") if cfg.use_resume else None



    #3.设置混合精度
    device_dtype = cfg.device
    dtype = torch.bfloat16 if cfg.dtype == "bfloat16" else torch.float16
    #通过 ContextManager 设置自动混合精度,提升训练速度和降低缓存
    autocast_ctx = nullcontext() if device_dtype == "cpu" else torch.cuda.amp.autocast(dtype=dtype)




    #4.配置wandb监控平台
    if cfg.use_resume and is_main_process() :
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "must" if wandb_id else None
        wandb_run_name = f"YuchenModel-PPO-Epoch-{cfg.epochs}-BS-{cfg.batch_size}-LR-{cfg.lr}"
        wandb.init(project = cfg.wandb_proj, name = wandb_run_name,id = wandb_id,resume = resume)




    #5.初始化PPO的四个模型及外置奖励模型
    base_weight = "reason" if cfg.is_reasoning else "full_sft"
    #Actor模型(当前正在被训练的模型)
    actor_model = YuchenModelCausalLLM(model_config)
    if cfg.use_compile:
        actor_model = torch.compile(actor_model)
        Logger("使用compile")

    #todo 1:old_actor模型
    #Old actor model(提供旧策略的概率用于重要性采样,不计算梯度)
    old_actor_model = None
    old_actor_model = old_actor_model.eval().requires_grad_(False)
    #Reference Model(监督基线,用于计算KL散度,防止Actor退化,不计算梯度)

    #todo 2:reference模型
    ref_model = None
    ref_model = ref_model.eval().requires_grad_(False)
    #critic_model(预测该状态能获得多少回报,独立更新权重)
    moe_suffix = 'moe' if cfg.use_moe else ""
    ckp = f"{cfg.save_dir}/{base_weight}_{cfg.hidden_size}{moe_suffix}.pth"
    state_dict = torch.load(ckp,map_location=cfg.device)
    critic_model = CriticModel(model_config)
    #Critic模型通常复用Actor模型(基座模型)的隐藏层权重作为热启动
    critic_model.load_state_dict(state_dict,strict=False)
    critic_model = critic_model.to(cfg.device)

    #Reward模型
    reward_model = AutoModel.from_pretrained(cfg.reward_model_path,torch_dtype=torch.bfloat16,trust_remote_code=True)
    reward_model = reward_model.to(cfg.device).eval().require_grad_(False)
    reward_tokenizer = AutoTokenizer.from_pretrained(cfg.reward_model_path,trust_remote_code=True)



    #6.数据和优化器配置
    train_ds = PPODataset(tokenizer,cfg)
    train_sampler = DistributedSampler(train_ds, shuffle=True) if dist.is_initialized() else None

    #critic_model 和 Actor_model分别使用独立的优化器
    actor_optimizer = optim.AdamW(actor_model.parameters(),lr=cfg.lr)
    critic_optimizer = optim.AdamW(critic_model.parameters(),lr=cfg.lr)

    #计算总迭代次数用来初始化学习率调度器
    load_for_count = DataLoader(train_ds,batch_size= cfg.batch_size,sampler= train_sampler)
    iters = len(load_for_count)
    total_optimizer_steps = (iters//cfg.accumulation_steps) * cfg.epochs

    actor_scheduler = CosineAnnealingLR(actor_optimizer,T_max=cfg.total_optimizer_steps,eta_min=cfg.lr//10)
    critic_scheduler = CosineAnnealingLR(actor_optimizer,T_max=cfg.total_optimizer_steps,eta_min=cfg.lr//10)


    #7.从checkpoint恢复状态(如果启用了resume)
    start_epoch,start_step = 0,0
    if ckp_data:
        actor_model.load_state_dict(ckp_data['model'])
        critic_model.load_state_dict(ckp_data['critic_model'])
        actor_optimizer.load_state_dict(ckp_data['optimizer'])
        critic_optimizer.load_state_dict(ckp_data['critic_optimizer'])
        actor_scheduler.load_state_dict(ckp_data['scheduler'])
        critic_scheduler.load_state_dict(ckp_data['critic_scheduler'])
        start_epoch = ckp_data['epoch']
        start_step = ckp_data.get('step', 0)



    #8.DDP模型包装
    if dist.is_initialized():
        #分布式包装Actor和Critic
        actor_model = DistributedDataParallel(actor_model,device_ids=[local_rank],output_device=local_rank)
        critic_model = DistributedDataParallel(critic_model,device_ids=[local_rank],output_device=local_rank)





    #9.主巡礼那循环
    for epoch in range(start_epoch,cfg.epochs):
        #短路求值,and运算符会先求左侧表达式train_sampler如果train_sampler为None
        #就不会执行右边的式子
        train_sampler and train_sampler.set_epoch(epoch)
        set_seed(42)
        #生成一个随机打乱的索引列表,用于后续打乱数据的顺序
        #torch.randperm(N)生成一个[0,N-1]的随机排列(即打乱顺序的整数张量)
        indices = torch.randperm(iters).tolist()


        #判断自己是否需要跳过已经训练的batch(断点续训)
        skip = start_step if (epoch == start_epoch) else 0
        batch_sampler = SkipBatchSimple(train_sampler or indices,batch=cfg.batch_size,skip_batch=skip)
        loader = DataLoader(train_ds,num_workers=cfg.num_worker,pin_memory=True,batch_sampler = batch_sampler)

        #调用PPO训练逻辑
        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{cfg.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            ppo_train_one_epoch(epoch=epoch, loader=loader, start_step=start_step, old_actor_model=old_actor_model,
                                ref_model=ref_model, actor_scheduler=actor_scheduler,
                                critic_scheduler=critic_scheduler, reward_model=reward_model,
                                reward_tokenizer=reward_tokenizer, wandb=wandb,cfg=cfg,iter=iters,critic_model=critic_model,
                                actor_model=actor_model,tokenizer=tokenizer)
        else:
            ppo_train_one_epoch(epoch=epoch, loader=loader, start_step=start_step, old_actor_model=old_actor_model,
                                ref_model=ref_model, actor_scheduler=actor_scheduler,
                                critic_scheduler=critic_scheduler, reward_model=reward_model,
                                reward_tokenizer=reward_tokenizer, wandb=wandb, cfg=cfg, iter=iters,
                                critic_model=critic_model,
                                actor_model=actor_model, tokenizer=tokenizer)

        #训练结束，清理分布式进程组
    if dist.is_initialized(): dist.destroy_process_group()
