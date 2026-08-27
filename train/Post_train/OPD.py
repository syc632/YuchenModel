import torch
import torch.nn as nn
from pathlib import Path
import json
import torch.nn.functional as f
from dataclasses import dataclass
from contextlib import nullcontext

from transformers import AutoTokenizer,AutoModelForCausalLM
from model.model import Config
from model.model import YuchenModelCausalLLM
from train.train_util import *


"""
OPD训练流程:
    同一个prompt --> 学生生成response --> prompt + response 同时送到教师模型和学生模型
    --> 在response的每个位置,取两者对"下一内容"的概率 --> 计算KL散度
    
关于如何投影到同一空间:
    教师模型把原本在教师词表中的概率分布转换为在候选文本上的概率分布
        例:     教师token ID    解码文本    logits
                1                hello       7.0
                2                world       6.0
                3                一          5.0
        先只在3个token上做softmax得到概率分布
                1               0.665
                2               0.245
                3               0.090
        然后把tokenID解码为文本
                hello               0.665
                world               0.245
                一                  0.090
    学生模型的流程就是根据老师模型生成的候选文本,预测每个位置的概率,然后在词表中找到相应的token概率并相加
        例:
"""




@dataclass
class OPDConfig:
    project_dir: Path = Path(r"...")
    data_file: str = ""
    tokenizer_dir: str = "BPEmodel"
    student_checkpoint: str = ""
    model_weight = ""
    save_path: str = ""

    teacher_model_path: Path = Path("...")

    max_samples: int | None = 1_000
    max_prompt_tokens: int = 384
    max_new_tokens: int = 128
    teacher_top_k: int = 8
    temperature: float = 1.0
    top_p: float = 0.95
    kd_temperature: float = 1.0
    kd_weight: float = 1.0
    moe_aux_weight: float = 1.0

    lr: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    epochs: int = 1
    gradient_accumulation_steps: int = 1
    min_lr_ratio: float = 0.1
    warmup_ratio: float = 0.03


    resume:bool = True
    use_compile = True
    log_interval: int = 10
    save_interval: int = 200
    seed: int = 42
    use_wandb: bool = True
    project_name: str = "YuchenModel_opd"

    use_moe: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    hidden_size: int = 512

    @property
    def data_path(self) -> Path:
        return self.project_dir / self.data_file

    @property
    def tokenizer_path(self) -> Path:
        return self.project_dir / self.tokenizer_dir

    @property
    def student_checkpoint_path(self) -> Path:
        return self.project_dir / self.student_checkpoint


@dataclass(frozen=True)
class TeacherStep:
    """用于盛放教师模型生成的蒸馏信息"""

    response_prefix: str
    candidate_texts: tuple[str, ...]
    teacher_log_probs: torch.Tensor


@dataclass
class StudentRollout:
    """
    用于盛放学生模型生成的采样信息
    """
    prompt_ids: torch.Tensor
    prompt_text:str
    response_ids: torch.Tensor
    response_text: str



def truncate_messages(messages, tokenizer, max_tokens):
    effective = [dict(message) for message in messages]

    while len(effective) > 1:
        rendered = render_chat_prompt(tokenizer, effective)
        if len(encode_text(tokenizer, rendered)) <= max_tokens:
            return effective

        # 删除最早的一轮消息
        effective.pop(0)

    rendered = render_chat_prompt(tokenizer, effective)
    if len(encode_text(tokenizer, rendered)) > max_tokens:
        raise ValueError("最后一条消息本身超过 max_prompt_tokens")

    return effective




def extra_prompt_message(conversation):
    """
    如果是SFT数据集,把答案去除,OPD通过学生的回答来采样,这个函数是去除单条数据的answer部分
    """
    message = [dict(message) for message in conversation]
    prompt = message[:-1]
    return prompt



def load_prompt(path:Path,max_sample=None):
    #把数据集中的回答部分删除掉,后面使用学生模型进行采样
    #此函数用于去除多条数据的answer部分
    prompt = []
    with path.open("r",encoding="utf-8") as f:
        #遍历数据集中的所有token
        for line_ids,line in enumerate(f):
            #如果当前行为空行,则跳过
            if not line.strip():
                continue
            #把json转换为字典
            item = json.loads(line)
            prompt.append(extra_prompt_message(item["conversations"]))
            if max_sample is not None and line_ids >= max_sample:
                break
    return prompt


def render_chat_prompt(tokenizer,messages):
    #用于把messages转换为模型要求的prompt格式
    return tokenizer.apply_chat_template(
        list(messages), tokenize=False, add_generation_prompt=True
    )


def encode_text(tokenizer, text):
    #用于使用tokenizer编码文字,文字 -> token id
    encoded = tokenizer(text, add_special_tokens=False)
    return list(encoded.input_ids)




def next_token_id(logits,temperature,top_p):

    if temperature <= 0:
        #温度为负数,贪心解码
        return logits.argmax(dim=-1, keepdim=True)
    #logits : b l vocab_size
    scaled = logits/temperature
    #对scaled进行降序排序,并返回排序后的logits和索引
    sorted_logits,sorted_indices = torch.sort(scaled,descending=True)
    #对sorted_logits进行softmax处理,得到概率分布
    sorted_prob = f.softmax(sorted_logits,dim=-1)
    #对sorted_prob计算累计和,并减去自身的数值,得到阈值
    remove = torch.cumsum(sorted_prob,dim=-1)
    remove = remove - sorted_prob > top_p
    #把达到top_p的logits设为False,因为masked_fill遮蔽True的位置
    remove[...,0] = False
    filtered = torch.full_like(scaled,-torch.inf)
    #把sorted_logits中达到top_p以上的logits按照索引sorted_indices映射到filtered中
    filtered.scatter_(dim=-1,index=sorted_indices,src=sorted_logits.masked_fill(remove,-torch.inf))
    #从filtered中随机抽取n_samples个logits,返回索引
    return torch.multinomial(f.softmax(filtered,dim=-1),num_samples=1)



@torch.inference_mode()
def sample_student_response(tokenizer,prompt,model,max_prompt_tokens,max_samples,temperature,top_p,device):
    """
    采样一条学生的轨迹并转为文本
    """
    #把prompt编码为token id
    prompt_ids = encode_text(tokenizer,prompt)
    if len(prompt_ids) > max_prompt_tokens:
        raise ValueError("传入 sample_student_response 的 prompt 尚未正确截断")
    input_ids = torch.tensor([prompt_ids],dtype = torch.long,device=device)

    #记录一下现在是训练模型还是评估模式,
    state = model.training
    #因为此函数是在训练循环中被调用,模型处于train状态,但是采样不希望被
    #Dropout或者RMSNorm干扰,不需要计算梯度,因此临时切到eval()
    model.eval()
    #预填充,把提示词先给model做一次前向传播
    generate = []
    output = model(input_ids=input_ids,logits_to_keep=1)
    cache = output.past_key_values
    for _ in range(max_samples):
        next_id = next_token_id(output.logits[:,-1,:],temperature=temperature,top_p=top_p)
        generate.append(next_id)
        if next_id.item() == tokenizer.eos_token_id:
            break
        output = model(input_ids=next_id,cache=cache,logits_to_keep = 1)
        cache = output.past_key_values
    model.train(state)
    response_ids = torch.cat(generate, dim=1)[0].tolist()
    response_text = tokenizer.decode(response_ids,skip_special_tokens=True)
    effective_prompt_ids = input_ids[0].tolist()
    rollout = StudentRollout(
        prompt_ids=effective_prompt_ids,
        prompt_text=prompt,
        response_ids=response_ids,
        response_text=response_text,
    )
    return rollout



def decoder_extension(tokenizer,prefix_ids,token_id):
    """
    通过两次解码的差集来获取单个token在特定上下文中的真实文本增量
    在BPE tokenizer中,很多token并不是一个完整的单词/字节,因此按照token_id进行解码会得到一个不完整的单词
    因此先把prefix与token_id拼接后解码,可以拿到完整的单词
    """
    pre_text = tokenizer.decode(prefix_ids,skip_special_tokens=True,clean_up_tokenization_spaces = False)
    full_text = tokenizer.decode(prefix_ids + [token_id], skip_special_tokens=True,clean_up_tokenization_spaces=False)
    return full_text[len(pre_text):]




class LocalTeacherModel:
    def __init__(self,model:nn.Module,tokenizer,top_k,device:torch.device):
        """
        冻结教师模型,为学生模型生成的每个token去打分

        """


        self.model = model.eval()
        self.tokenizer = tokenizer
        self.device = device
        self.top_k = top_k


        for param in model.parameters():
            param.requires_grad_(False)

    @torch.inference_mode()
    def response_score(self,messages,response,temperature):
        """

        :param messages: 问题
        :param response: 回答
        :param temperature:
        :return:
        """

        #1.先用教师自己的tokenizer构造输入
        #把提示词转换为模型要求的prompt格式
        prompt = render_chat_prompt(tokenizer=self.tokenizer,messages=messages)
        #编码提示词和回答,生成一个1维的索引序列
        prompt_ids = encode_text(self.tokenizer,prompt)
        response_ids = encode_text(self.tokenizer,response)




        #2.教师模型对整段prompt+response 进行一次前向传播,一次性得到回答的每个位置的概率分布
        #1,l []相当于是给prompt_ids+response_ids升维,因为HuggingFace的模型接口必须是二维的
        input_ids = torch.tensor([prompt_ids+response_ids],device=self.device,dtype=torch.long)
        #模型的输出一个序列的logits,即prompt+response序列中每个token对于下一个token的预测分布
        logits = self.model(input_ids).logits[0]         #只有一个样本,batch=1,模型输出完用[0]把batch这个维度消掉即可
        step = []



        #3.遍历学生回答在教师词表中的每个token位置
        for token_index in range(len(response_ids)):
            #在prompt+当前回答前缀之后,教师预测的下一个token的logits
            next_logits = logits[len(prompt_ids) + token_index - 1]



        #4.选词表中的前k个概率作为老师的打分
            k = min(self.top_k, len(next_logits))
            value,ids = torch.topk(next_logits/temperature,k=k)
            #用来装教师模型在某个token上的top-k候选文本及其logits
            #候选文本为在每个token位置上教师模型的词表中logits最大的k个token_id所对应的文本,logits为这些文本对应的logits
            candidates = {}
            prefix_ids = prompt_ids + response_ids[:token_index]




        #5.把老师的token转为文本续写
            for value,token_id in zip(value,ids):
                #把logits编码为文本,因为学生教师词表不一样,所以先解码为文字
                txt_extend = decoder_extension(self.tokenizer,prefix_ids,token_id)




        #6.把编码好的文字放入到candidates
                #由于后续投影空间是文本，相同文本对应的所有教师 token 概率应使用 torch.logaddexp 聚合。
                if txt_extend in candidates:
                    candidates[txt_extend] = torch.logaddexp(candidates[txt_extend], value)
                else:
                    candidates[txt_extend] = value
            #如果只有一个候选的化,后续经过softmax这个token的概率为100%
            #计算KL散度的时候Loss为0,因此梯度也为0
            #当candidates的长度小于2的时候跳过即可
            if len(candidates) < 2:
            #当多个不同的token ID解码为相同的文本或者解码出多个空字符串的时候可能出现候选文本为1的情况
                continue
            #取出字典的键,即候选文本字符,并冻结为一个不可变得元组
            #这些文字符串会被学生的tokenizer重新编码,因为学生教师词表不一样,所以需要用文本去做桥梁
            candidates_text = tuple(candidates)
            #取出字典的logits
            candidates_logits = torch.stack([candidates[text] for text in candidates_text],dim=0)
            teacher_log_probs = f.log_softmax(candidates_logits,dim=0)
            prefix_text = self.tokenizer.decode( prefix_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
            step.append(TeacherStep(response_prefix=prefix_text,candidate_texts=candidates_text,teacher_log_probs=teacher_log_probs))
        return step




def longest_common_prefix(left,right):
    """
    用于找到两个序列的最大公共前缀长度
    """
    index = 0
    while index< len(left) and index < len(right) and right[index] == left[index]:
        index += 1
    return index


def score_student(model,tokenizer,student_prompt,response_prefix,candidates,padding_token_id,device):
    """
    把教师的若干 top-k 候选文本”转换成学生模型可比较的分数
    候选文本可能对应学生词表的多个token,因此概率是这些token的联合log之和
    例:
        已有回答前缀:今天天气
        教师模型候选A: 很好
        教师模型候选B: 不错
        教师模型可能把"很好"和"不错"视为一个token,但是学生可能把"很" 视为一个token,"好"视为一个token
        因此函数要算的是联合概率 : logP(很|今天天气) + logP(好|今天天气)

    """
    #1.编码公共文本前缀
    base_id = encode_text(tokenizer=tokenizer,text=(student_prompt+response_prefix))



    #2.拼上教师模型给的候选文本,并用学生的tokenizer进行分词,这样每个候选文本都能得到一个学生序列
    #candidates_id为列表套列表的形式,里面的列表是多个候选文本id
    # 列表推导式,立即算出所有的结果,存成列表,后面可复用
    candidates_id = [encode_text(tokenizer=tokenizer,text=(student_prompt+response_prefix+candidate)) for candidate in candidates]




    #3.找最长的公共前缀长度,因为文本拼接处可能会出现重切分的现象
    #即追加文本后,原来前缀的末尾token可能会和追加文本的开头token粘在一起:
    #例:
    #   源文本: "Hello"
    #   追加文本:"World"
    #   拼接后:"HelloWorld" BPE分词器把源文本和追加的文本切分为同一个token了
    start = [longest_common_prefix(base_id,ids )for ids in candidates_id]
    #map映射到candidates_id列表中的每个列表,这样可以得到最长的列表长度
    max_len = max(map(len,candidates_id))




    #4.把所有候选文本padding成一个batch
    #创建两个全0的画布,然后再用数值填充,达到padding和mask的效果
    input_ids = torch.full((len(candidates_id),max_len),padding_token_id,dtype=torch.long,device=device)
    mask = torch.zeros_like(input_ids,dtype=torch.bool)
    #row:行号,ids:每一行的完整token_id列表,也就是token_id
    for row,ids in enumerate(candidates_id):
        input_ids[row,:len(ids)] = torch.tensor(ids,dtype=torch.long,device=device)
        mask[row,:len(ids)] = True


    #5.学生前向传播,并获取序列中每个位置的词表logits
    output = model(input_ids = input_ids,attention_mask = mask)
    log_prob = f.log_softmax(output.logits,dim=-1)



    #6.取出每条候选新增部分的概率并求和
    score = []
    for row,(ids,start) in enumerate(zip(candidates_id,start)):
        #生成一个一维索引用于target_ids
        token_position = torch.arange(start,len(ids),dtype=torch.long,device=device)
        #位置t由t-1生成
        predicted_by = token_position-1
        #高级索引,索引的张量每个元素取出一个值
        target_ids = input_ids[row,token_position]
        #取出新增文本在词表中对应的log概率求和
        #得到每个候选序列的概率
        score.append(log_prob[row,predicted_by,target_ids].sum())
        #score是一个python列表,里面k个标量,torch.stack把列表堆叠为一个((k,))的张量
    return torch.stack(score),output.aux_loss




def projected_top_k_kl(student_model,student_tokenizer,response,teacher_step,pad_token_id,device):
    """
    计算一个位置上的KL(教师top-k || 学生投影top-K)
    """
    student_score,aux_loss = score_student(model=student_model,
                                  tokenizer=student_tokenizer,
                                  student_prompt=response.prompt_text,
                                  response_prefix=teacher_step.response_prefix,
                                  candidates=teacher_step.candidate_texts,
                                  padding_token_id=pad_token_id,
                                  device=device)
    student_log_prob = f.log_softmax(student_score.float(),dim=0)
    teacher_prob = teacher_step.teacher_log_probs.exp().to(dtype = student_log_prob.dtype,device=device)
    #KL(q||p) = Σ q_i (logq_i - logp_i)
    return f.kl_div(student_log_prob,teacher_prob,reduction="sum"),aux_loss




def opd_loss_rollout(student_model,student_tokenizer,response,teacher_step,pad_token_id,device):
    """
    计算一条序列上kl_loss和aux_loss的平均
    """
    kl_terms = []
    aux_terms = []
    for step in teacher_step:
        kl,aux = projected_top_k_kl(student_model=student_model,student_tokenizer=student_tokenizer,response=response,
                                    teacher_step=step, pad_token_id=pad_token_id, device=device)
        kl_terms.append(kl)  #列表套tensor [tensor(0.123),tensor(0.456).....]
        aux_terms.append(aux)

    #当学生只生成EOS、教师没有有效候选时,torch.stack会报错,提前处理
    if not kl_terms:
        zero = next(student_model.parameters()).new_zeros(())
        return zero, zero, 0
    #torch.stack把已有张量拼接:tensor[0.123,0.456,....]
    return torch.stack(kl_terms).mean(),torch.stack(aux_terms).mean(),len(kl_terms)




def train_one_epoch(epoch,start_step,iter,update,accepted,student_tokenizer,opd_config,wandb=None):
    start_time = time.time()
    for step in range(start_step,iter):
        message = prompt[step]
        effective_message = truncate_messages(messages=message, tokenizer=student_tokenizer, max_tokens=opd_config.max_prompt_tokens)
        student_prompt = render_chat_prompt(tokenizer=student_tokenizer, messages=effective_message)
        #学生采样
        response = sample_student_response(tokenizer=student_tokenizer, prompt=student_prompt, model=student_model,
                                           max_prompt_tokens=opd_config.max_prompt_tokens,
                                           max_samples=opd_config.max_new_tokens,
                                           temperature=opd_config.temperature, top_p=opd_config.top_p,
                                           device=device)
        #教师打分
        teacher_step = teacher.response_score(messages=effective_message, response=response.response_text, temperature=opd_config.kd_temperature)


        global_step = epoch*iter + step
        total_step = opd_config.epochs*iter

        lr = get_lr(current_step=global_step, total_step=total_step,lr=opd_config.lr,warmup_ratio=opd_config.warmup_ratio,min_lr_ratio=opd_config.min_lr_ratio)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr



        with auto_cast:
            #算损失
            kl_loss, aux_loss, valid = opd_loss_rollout(student_model, student_tokenizer, response, teacher_step, student_tokenizer.pad_token_id,device)

            if valid == 0:
                continue

        loss = opd_config.kd_weight * kl_loss + opd_config.moe_aux_weight * student_model.aux_loss_alpha * aux_loss
        (loss / opd_config.gradient_accumulation_steps).backward()


        torch.nn.utils.clip_grad_norm_(student_model.parameters(), opd_config.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        update += 1
        accepted += 1

        #打印日志
        if update % opd_config.log_interval == 0:
            spend_time = time.time() - start_time
            current_lr = optimizer.param_groups[0]["lr"]
            eta_time = spend_time / (step + 1) * iter // 60 - spend_time // 60
            Logger(
                f"轮数:{epoch}/{opd_config.epochs}"
                f",当前步数:{step},KL损失:{kl_loss.item():.4f},"
                f"辅助损失:{aux_loss.item():.4f},KL散度:{kl_loss.item():.4f},学习率:{current_lr},剩余时间:{eta_time}分钟")
            if wandb is not None:
                wandb.log({
                    "loss": loss.item(),
                    "aux_loss": aux_loss.item(),
                    "logits_loss": kl_loss.item(),
                    "lr": current_lr,
                    "eta_time": eta_time
                })

        if update % opd_config.save_interval == 0:
            lm_check_point(lm_config=opd_config, weight=opd_config.model_weight, model=student_model, optimizer=optimizer,step=step+1,
                           update=update,accepted=accepted)

        # 保存模型
        if (step % opd_config.save_interval == 0 or step == iter - 1) and is_main_process():
            lm_check_point(lm_config=opd_config, weight=opd_config.model_weight, model=student_model,optimizer=optimizer,
                           epoch=epoch+1,step=0,update = update,accepted = accepted)

    optimizer.zero_grad(set_to_none=True)

    del message, loss
    return update,accepted



if __name__ == "__main__":
    # 流程
    # 创建模型→ 创建优化器→ 加载模型checkpoint→ 加载优化器checkpoint→ torch.compile→ 开始训练


    opd_config = OPDConfig()
    model_config = Config()
    set_seed(seed=opd_config.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"


    #student and teacher model
    student_model = YuchenModelCausalLLM(model_config)
    student_tokenizer = AutoTokenizer.from_pretrained(opd_config.tokenizer_path)
    student_model.load_state_dict(torch.load(opd_config.student_checkpoint_path, map_location="cpu", weights_only=True))
    student_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    student_model = student_model.to(device, dtype=student_dtype)
    teacher_tokenizer = AutoTokenizer.from_pretrained(opd_config.teacher_model_path, trust_remote_code=True)
    teacher_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    teacher_model = AutoModelForCausalLM.from_pretrained(opd_config.teacher_model_path, dtype=teacher_dtype,trust_remote_code=True).to(device)
    teacher = LocalTeacherModel(teacher_model, teacher_tokenizer, opd_config.teacher_top_k, device=device)


    prompt = load_prompt(opd_config.data_path, opd_config.max_samples)
    optimizer = torch.optim.AdamW(student_model.parameters(), lr=opd_config.lr, weight_decay=opd_config.weight_decay)
    optimizer.zero_grad(set_to_none=True)

    auto_cast = torch.amp.autocast(device, dtype=torch.bfloat16) if device == "cuda" else nullcontext()
    #加载OPD训练断点
    ckp_data = lm_check_point(lm_config=opd_config, weight=opd_config.model_weight,optimizer=optimizer) if opd_config.resume else None


    # wandb
    wandb_run = None
    if opd_config.use_wandb:
        wandb_id = ckp_data.get("wandb_id") if ckp_data else None
        resume = "allow" if wandb_id else None
        wandb_run_name = f"YuchenModel-OPD-Epoch-{opd_config.epochs}-LearningRate-{opd_config.lr}"
        wandb_run = wandb.init(project=opd_config.project_name, id=wandb_id, name=wandb_run_name,resume=opd_config.resume)


    if ckp_data:
        student_model.load_state_dict(ckp_data["model_state"])
        optimizer.load_state_dict(ckp_data["optimizer_state"])


    #compile
    if opd_config.use_compile:
        student_model = torch.compile(student_model)#图编译加速器
        Logger("使用compile")

    start_epoch = 0
    start_step = 0
    update = 0
    accepted = 0

    for epoch in range(opd_config.epochs):
        set_seed(opd_config.seed + epoch)
        skip = start_step if epoch == start_epoch else 0


        if skip > 0:
            Logger(f'Epoch [{epoch + 1}/{opd_config.epochs}]: 跳过前{start_step}个step，从step {start_step + 1}开始')
            update, accepted = train_one_epoch(epoch=epoch, iter=len(prompt), update=update, accepted=accepted,student_tokenizer=student_tokenizer,
                                               opd_config=opd_config,start_step=skip)
        else:
            update, accepted = train_one_epoch(epoch=epoch, iter=len(prompt), update=update, accepted=accepted,student_tokenizer=student_tokenizer,
                                               opd_config=opd_config, start_step=skip)
