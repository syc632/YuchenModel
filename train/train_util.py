from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist  #分布式训练
import torch
import random
import os
import time
import math
from torch.utils.data import DataLoader,Dataset,Sampler
try:
    import wandb  #日志
except ImportError:
    wandb = None

try:
    from datasets import load_dataset  #数据加载
except ImportError:
    load_dataset = None




#先判断是否多卡分布式训练,没有启动 == > 直接print/ 已经启动 ==> 判断当前进程rank是否为0
#rank = 0 ==>print   rank 不等于0 ==>什么都不做
def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0

def Logger(content):
    #主程序使用
    if is_main_process():
        print(content)


#随机种子
def set_seed(seed:int=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def retry(src,trg,attmpt=10,delay=1):
    """
    原子替换的时候如果进程被占用(如偶发病毒扫描,文件锁)
    :param src:临时文件
    :param trg:目标文件
    :param attmpt: 尝试次数
    :param delay: 延迟时间
    """
    for i in range(attmpt):
        try:
            os.replace(src,trg)
            return
        except PermissionError:
            if i == attmpt - 1:
                raise
            time.sleep(delay)



def lm_check_point(lm_config,weight:str=None,model=None,optimizer=None,epoch=None,step=0,**kwargs):
    """
    kwargs: 用于保存额外的训练状态
    """
    save_dir = lm_config.save_path
    #检查/创建文件
    os.makedirs(save_dir,exist_ok=True)
    #根据是否使用Moe来构造文件名
    moe_path = "_moe" if lm_config.use_moe else ""
    #保存模型权重,用于推理
    check_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}.pth"
    #完整的训练断点文件的路径,用于中断后恢复训练
    resume_path = f"{save_dir}/{weight}_{lm_config.hidden_size}{moe_path}_resume.pth"

    #模式A:保存check_point
    if model is not None:
        #去掉训练包装后的原始模型,正确取得模型参数
        #训练中model可能不是原始模型,而是被DDP或torch.compile包装过的,raw_model就是拿到原始的模型
        raw_model = model.module if isinstance(model,DistributedDataParallel) else model
        raw_model = getattr(raw_model,"_orig_mod",raw_model)
        #state_dict()用来获取模型或优化器当前的状态
        #state_dict是模型内部的"参数名:参数数值"字典,保存的是模型在训练中学习到的数值
        state_dict = raw_model.state_dict()
        #转为半精度并移到CPU,节省磁盘空间并且不占用显存
        state_dict = {
            k:(v.half() if v.is_floating_point() else v).cpu()
            for k,v in state_dict.items()
        }

        #原子保存,先写临时文件,确认临时文件完整写入,再替换旧文件
        ckp_tmp = check_path+".tmp"
        torch.save(state_dict,ckp_tmp)
        retry(ckp_tmp, check_path)

        wandb_id = None
        #获取WandB的run_id,确保重启训练后日志可以接上
        if wandb is not None and wandb.run is not None:
            wandb_id = wandb.run.id
        resume_data = {
            "model": state_dict,  #模型参数
            "optimizer": optimizer.state_dict() if optimizer is not None else None, #优化器状态
            "epoch": epoch,
            "step": step,
            "wandb_id": wandb_id,
            "world_size":dist.get_world_size() if dist.is_initialized() else 1,
        }

        #处理额外需要保存的对象,例如Scaler和当前更新步数
        for k,v in kwargs.items():
            if v is not None:
                resume_data[k] = v.state_dict() if hasattr(v,"state_dict") else v
        resume_tmp = resume_path+".tmp"
        torch.save(resume_data,resume_tmp)
        retry(resume_tmp, resume_path)

        #显式删除较大的临时对象并清理显存缓存
        del state_dict, resume_data
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    #模式B,加载checkpoint
    else:
        if os.path.exists(resume_path):
            #将数据加载到CPU,避免模型和checkpoint同时驻留GPU导致峰值显存超限
            ckp_data = torch.load(resume_path, map_location=torch.device('cpu'))
            #处理GPU数量变化后的step转换,尽量保持已经消费的数据量一致
            saved_ws = ckp_data.get('world_size', 1)
            current_ws = dist.get_world_size() if dist.is_initialized() else 1
            if saved_ws != current_ws:
                ckp_data['step'] = ckp_data['step'] * saved_ws // current_ws
                Logger(f'GPU数量变化({saved_ws}→{current_ws})，step已自动转换为{ckp_data["step"]}')
            return ckp_data
        return None






class SkipBatchSimple(Sampler):
    def __init__(self,sampler,batch,skip_batch=0):
        """
        初始化断点续采样器
        例如:在第1000批的时候中断了,断点续采样器会从第1001批的数据交给模型
        :param sampler: 基础采样器,例如SequentialSampler
        :param batch: 每个batch包含的样本数量
        :param skip_batch: 计划跳过多少个batch
        """
        self.sampler = sampler
        self.batch = batch
        self.skip_batch = skip_batch

    def __iter__(self):
        batch = [] #用于缓存当前的batch索引
        skipped = 0 #记录已经跳过的batch数量
        for idx in self.sampler:
            batch.append(idx)
            if len(batch) == self.batch:
                #如果还没有跳过足够数量的batch,清空当前batch继续遍历
                if skipped < self.skip_batch:
                    skipped+=1
                    batch = []
                else:
                    yield batch #把当前一批次的索引交给DataLoader
                    batch = []  #为下一批清空旧索引
        #如果样本总数不能被batch整除,处理最后剩余的样本
        if len(batch) > 0 and skipped >= self.skip_batch:
            yield batch

    def __len__(self):
        """计算跳过指定batch后还剩余多少个batch。"""
        total_batch = (len(self.sampler)+self.batch-1) // self.batch
        return max(0,total_batch-self.skip_batch)


def get_model_params(model, config):
    # 1.计算总参数量,除以1e6转换为百万(M)
    total = sum(p.numel() for p in model.parameters()) / 1e6
    # 2.统计所有层中路由专家的总参数量
    route_total = sum(
        p.numel() for n, p in model.named_parameters()
        if ".ffn.route_expert." in n
    ) / 1e6
    # 3.统计所有层中共享专家的总参数量
    shared_total = sum(
        p.numel() for n, p in model.named_parameters()
        if ".ffn.shared_expert." in n
    ) / 1e6

    if config.use_moe and config.n_route_expert > 0:
        # 4.骨架参数包含Embedding、Mixer、Norm、LM Head等公共参数
        base = total - route_total - shared_total
        # 5.每个token只激活top-k个路由专家,所有共享专家始终激活
        active = (
                base
                + route_total * config.n_expert_per_token / config.n_route_expert
                + shared_total
        )
        Logger(f'Model Params: {total:.2f}M-A{active:.2f}M')
    else:
        Logger(f'Model Params: {total:.2f}M')


def get_lr(current_step,total_step,lr,warmup_ratio,min_lr_ratio):
    """
    前期warmup从较小值升到最大值,后期按照余弦曲线逐渐降到最小学习率
    训练刚开始的时候优化器和学习率还不稳定,warmup先平缓升高,之后余弦退火
    """
    warmup_step = max(1,int(warmup_ratio*total_step))
    #warmup:从较小学习率线性升到lr
    if current_step < warmup_step:
        return lr*(current_step+1)/warmup_step
    #cosine:从lr逐渐降低到lr*min_lr_ratio
    progress = (current_step-warmup_step)/max(1,total_step-warmup_step)
    progress = min(progress, 1.0)
    return lr*(min_lr_ratio+(1-min_lr_ratio)*0.5*(1+math.cos(math.pi*progress)))




