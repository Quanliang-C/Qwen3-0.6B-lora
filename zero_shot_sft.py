import pandas as pd
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorWithPadding
from src.data_loader import load_data, build_prompt_applier, build_messages
from src.zero_shot_parsers import lc_parser, pyd_parser, pyd_format
from src.metric import calculate_metrics, print_metrics
from torch.utils.data import DataLoader
import torch
from tqdm.auto import tqdm
import argparse
from accelerate import Accelerator
from peft import PeftModel


def evaluate_chunk(generated_texts, true_labels, generated_lengths, chunk_number):
    """对一个数据块进行解析和评估，并打印中间结果和token信息"""
    if not generated_texts:
        return [], []

    print(f"\n----- Evaluating Chunk {chunk_number} ({len(generated_texts)} samples) -----")
    
    max_tokens_in_chunk = max(generated_lengths) if generated_lengths else 0
    print(f"Max generated tokens in this chunk: {max_tokens_in_chunk}")

    pred_labels = []
    for text in tqdm(generated_texts, desc=f"Parsing Chunk {chunk_number}", leave=False):
        try:
            result = pyd_parser.parse(text)
            pred_labels.append(result.label)
        except Exception:
            pred_labels.append(-1)

    valid_indices = [i for i, pred in enumerate(pred_labels) if pred != -1]
    valid_true = [true_labels[i] for i in valid_indices]
    valid_pred = [pred for pred in pred_labels if pred != -1]
    
    if valid_pred:
        metrics = calculate_metrics(valid_true, valid_pred)
        print(f"----- Chunk {chunk_number} Metrics -----")
        print_metrics(metrics)
        parse_rate = len(valid_pred) / len(pred_labels)
        print(f"Parsing Success Rate: {len(valid_pred)}/{len(pred_labels)} = {parse_rate:.4f}")
    else:
        print(f"All predictions in Chunk {chunk_number} failed to parse.")
    print("----------------------------------------\n")
    
    return valid_pred, valid_true


##  避免num_worker > 0产生的spawn调度问题
if __name__ == "__main__":

    has_cuda_gpu = torch.cuda.is_available()

    if has_cuda_gpu:
        capability = torch.cuda.get_device_capability()
        print(f"Detected GPU with Compute Capability: {capability[0]}.{capability[1]}")

        if capability[0] >= 8:
            # Ampere 架构 (A100, L4, RTX 30xx/40xx) 或更新的架构
            print("GPU supports bfloat16 hardware acceleration. Using bf16.")
            mixed_precision = "bf16"
            torch_type = torch.bfloat16
        else:
            # Turing 架构 (T4) 或更早的架构
            print("GPU does not have native bfloat16 support. Falling back to fp16.")
            mixed_precision = "fp16"
            torch_type = torch.float16
    else:
        # Mac (MPS) 或 CPU 环境
        print("No CUDA GPU detected. Using float32 without mixed precision.")
        mixed_precision = "no"
        torch_type = torch.float32


    accelerator = Accelerator(mixed_precision=mixed_precision)





    parser = argparse.ArgumentParser()
    parser.add_argument("--think", action="store_true")
    parser.add_argument("--load_batch_size", type=int, default=32)
    parser.add_argument("--eval_batch_size", type=int, default=16)

    args = parser.parse_args()

    with accelerator.main_process_first():
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B", padding_side="left")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            torch_dtype=torch_type,
            device_map=None
            ).eval()
        

        model_with_lora = PeftModel.from_pretrained(model, "sft_without_cot_final")


        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        _, test_dataset, _ = load_data()


        prompt_applier = build_prompt_applier(tokenizer, enable_thinking=args.think)


        prompt_inputs = test_dataset.map(
            prompt_applier,
            batched=True,
            batch_size=args.load_batch_size
        )

        prompt_inputs = prompt_inputs.remove_columns(["sentence1", "sentence2"])
        prompt_inputs = prompt_inputs.rename_column("label", "metric_label")

        ## 交给dataloader处理
        input_tokenized_datasets = prompt_inputs.map(
            lambda x: tokenizer(x['text'], truncation=True, padding=False),
            batched=True,
            batch_size=args.load_batch_size
        )

        ## 对每个样本计算token长度，排序，对GPU并行计算友好，每个batch的样本长度相近
        input_tokenized_datasets = input_tokenized_datasets.map(
            lambda x: {'length': len(x['input_ids'])},
            batched=False
        )
        input_tokenized_datasets = input_tokenized_datasets.sort('length')
        input_tokenized_datasets = input_tokenized_datasets.remove_columns(["text", "length"])



    dataloader = DataLoader(
        input_tokenized_datasets,
        batch_size=args.eval_batch_size,
        shuffle=False,
        num_workers=8,
        collate_fn=DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt", padding=True))
    ## pin_memory 方便数据传输到gpu


    model, dataloader = accelerator.prepare(model, dataloader)


    ## 打印模型数据类型，确保模型运行在正确的数据类型上
    if accelerator.is_main_process:
        unwrapped_model = accelerator.unwrap_model(model)
        model_dtype = next(unwrapped_model.parameters()).dtype
        print(f"----------------------------------------")
        print(f"Model is running with data type: {model_dtype}")
        print(f"----------------------------------------")




    total_valid_pred = []
    total_valid_true = []
    total_samples_processed = 0


    num_batch = len(dataloader)

    for batch_idx, batch in tqdm(enumerate(dataloader, start=1), total=num_batch, disable=not accelerator.is_main_process, desc="Generating"):
        with torch.no_grad():
            unwrapped_model = accelerator.unwrap_model(model)
            input_ids_len = batch["input_ids"].shape[1]
            generated_ids = unwrapped_model.generate(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                max_new_tokens=4096,
                temperature=0.1,
                do_sample=True
            )
        

        
            # 从每个生成序列中，只截取新生成的部分
            # generated_ids 的形状是 [batch_size, input_len + new_len]
            newly_generated_ids = generated_ids[:, input_ids_len:]
        
        ## 其实不需要.cpu()，tolist() 会自动转成cpu
        newly_generated_ids_cpu_list = newly_generated_ids.tolist()



        token_to_decode = []
        for seq in newly_generated_ids_cpu_list:
            try:
                first_content_index = len(seq) - seq[::-1].index(151668)
                token_to_decode.append(seq[first_content_index:])
            except ValueError:
                token_to_decode.append(seq)

        generated_texts = tokenizer.batch_decode(token_to_decode, skip_special_tokens=True)
        generated_lengths = [len(seq) for seq in newly_generated_ids_cpu_list]

        
    ### 存在问题
        # 这个是decode后的文本，用gather_for_metrics, 因为不是tensor
        # gather_for_metrics 会自动判断use_gather_object(bool)，调用gather_object
        # gather_object 会递归的展开所有list，dict
        gathered_generated_texts = accelerator.gather_for_metrics(generated_texts)
        gathered_true_labels = accelerator.gather_for_metrics(batch["metric_label"])
        # 先转到cpu decode, 再转回tensor, 利用显卡的高效通信分发，也许效率更高？暂时保留cpu传输
        # 我不确定gather_for_metrics 是否掉用了
        gathered_generated_lengths = accelerator.gather_for_metrics(generated_lengths)
    ### 存在问题



        ## 文本是本轮的，用完就丢了

        if accelerator.is_main_process:
            # ## list of list 转成 list, 暂时不需要，gather_object 会自动展开
            # gathered_generated_texts = [item for sublist in gathered_generated_texts for item in sublist]

            ## 每轮结束，评估一次，加入total，只保留指标
                
            valid_pred, valid_true = evaluate_chunk(gathered_generated_texts, gathered_true_labels.tolist(), gathered_generated_lengths, batch_idx)
            total_valid_pred.extend(valid_pred)
            total_valid_true.extend(valid_true)
            total_samples_processed += len(gathered_generated_texts)
            
            
            if batch_idx == num_batch:
                ## 存了最后一轮的主进程的ids lsit
                last_batch_newly_generated_ids_list = newly_generated_ids_cpu_list
                # 保存最后一批的raw output, 不删除think content，用于观察
                last_batch_raw_output = tokenizer.batch_decode(last_batch_newly_generated_ids_list, skip_special_tokens=True)
                if last_batch_raw_output:
                    print("example from last batch form main process:")
                    print(last_batch_raw_output[0])
                    print("--------------------------------")
                    try:
                        with open("last_batch_output.txt", "w", encoding="utf-8") as f:
                            for line in last_batch_raw_output:
                                f.write(line + "\n")
                        print("\nSuccessfully saved the last batch raw output to 'last_batch_output.txt'")
                    except Exception as e:
                        print(f"\nFailed to save last batch output to file: {e}")


    if accelerator.is_main_process:

        print("--------------------------------")
        print("final total eval start...")

        if total_valid_pred:
            metrics = calculate_metrics(total_valid_true, total_valid_pred)
            print_metrics(metrics)
            print(f"Parsing Success Rate: {len(total_valid_pred)}/{total_samples_processed} = {len(total_valid_pred)/total_samples_processed:.4f}")
        else:
            print("所有预测都解析失败，无法计算指标")