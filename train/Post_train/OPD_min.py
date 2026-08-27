import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import AutoTokenizer,AutoModelForCausalLM
device = "cuda" if torch.cuda.is_available() else "cpu"
"""
此代码用于示范OPD的运行流程,属于教学代码,并不适合实际使用
"""

student_name = "Qwen/Qwen3-0.6B"
teacher_name = "Qwen/Qwen3-1.7B"

#使用HuggingFace的Tokenizer和模型示范
tokenizer = AutoTokenizer.from_pretrained(student_name)

student = AutoModelForCausalLM.from_pretrained(student_name,dtype=torch.bfloat16).to(device)
teacher = AutoModelForCausalLM.from_pretrained(teacher_name,dtype=torch.bfloat16).to(device)

#教师模型不更新梯度
teacher.eval()

for p in teacher.parameters():
    p.requires_grad = False

optimizer = torch.optim.AdamW(student.parameters(),lr = 1e-5)

def opd_step(prompt):
    #B L
    inputs = tokenizer(prompt,return_tensors="pt").to(device)
    #print(f"inputs: {inputs}")
    prompt_len = inputs.input_ids.size(-1)

    #采样(不反传)
    with torch.no_grad():
        generated = student.generate(**inputs,max_new_tokens = 2,do_sample= True,temperature = 1.0,top_p = 0.95)


    #学生对自己的采样打分(反传)
    student_output = student(input_ids = generated)
    #logits和Lable错开一位
    student_logits = student_output.logits[:,:-1,:]


    #教师打分(不反传)
    with torch.no_grad():
        teacher_output = teacher(input_ids = generated)
        teacher_logits = teacher_output.logits[:,:-1,:]



    student_log_prob = f.log_softmax(student_logits,dim=-1)
    teacher_log_prob = f.log_softmax(teacher_logits,dim=-1)
    teacher_prob = teacher_log_prob.exp()



    start = prompt_len - 1
    teacher_log_prob = teacher_log_prob[:,start:]
    student_log_prob = student_log_prob[:,start:]
    teacher_prob = teacher_prob[:,start:]


    #KL散度
    kl_per_token = torch.sum(teacher_prob*(teacher_log_prob-student_log_prob),dim=-1)

    loss = kl_per_token.mean()


    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()

if __name__ == "__main__":
    prompt = "你是谁"
    opd_step(prompt)