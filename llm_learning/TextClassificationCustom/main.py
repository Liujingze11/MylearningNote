import os
import numpy as np
import json
import argparse
from utils_data import load_data, MyDataset # 从自定义数据处理文件中导入数据加载函数和数据集类
from modeling_bert import BertForSequenceClassification

# 从自定义数据处理文件中导入数据加载函数和数据集类
from transformers import (
    AutoConfig,                     # 自动加载模型配置
    BertForSequenceClassification,  # BERT文本分类模型
    AutoTokenizer,                  # 自动加载对应分词器
    EvalPrediction,                 # 评估时预测结果的数据结构
    Trainer,                        # Hugging Face 高级训练器
    TrainingArguments,              # 训练参数配置类
    default_data_collator,          # 默认 batch 拼接函数
    set_seed,                       # 设置随机种子，保证实验可复现
)

def parse_args():
    """
    解析命令行参数
    例如：
    python train.py --model bert-base-uncased --epoch 3
    """  
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--model', type=str, default='bert-base-uncased')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--epoch', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--bs', type=int, default=32)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--seed', type=int, default=666, help='random seed')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    
    # Set seed before initializing model, for reproduction purpose.
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    config = AutoConfig.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model =BertForSequenceClassification.from_pretrained(args.model, config=config)

    # Load data
    train_data = load_data(args, "train")
    train_dataset = MyDataset(train_data, tokenizer, args.max_length, is_test=False)
    eval_data = load_data(args, "val")
    eval_dataset = MyDataset(eval_data, tokenizer, args.max_length, is_test=False)
    test_data = load_data(args, "test")
    test_dataset = MyDataset(test_data, tokenizer, args.max_length, is_test=True)

    def compute_metrics(p: EvalPrediction):
        preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
        preds = np.argmax(preds, axis=1)
        correct = ((preds == p.label_ids).sum()).item()
        return {'accuracy': 1.0*correct/len(preds)}

    training_args = TrainingArguments(
            output_dir = args.output_dir,
            do_train=True,
            do_eval=True,
            do_predict=True,
            logging_strategy="steps",
            save_strategy="epoch",
            learning_rate= args.lr,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.bs,
            num_train_epochs=args.epoch,
            report_to="none"
        )
    
    # Initialize our Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=default_data_collator,
    )

    # Training
    if training_args.do_train:
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.save_model()  # Saves the tokenizer too for easy upload
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    if training_args.do_eval:
        metrics = trainer.evaluate(eval_dataset=eval_dataset)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        predictions = trainer.predict(test_dataset, metric_key_prefix="predict").predictions
        predictions = np.argmax(predictions, axis=1)
        output_predict_file = os.path.join(args.output_dir, "predict_results.txt")
        if trainer.is_world_process_zero():
            with open(output_predict_file, "w") as writer:
                writer.write("index\tprediction\n")
                for index, item in enumerate(predictions):
                    writer.write(f"{index}\t{item}\n")

if __name__ == "__main__":
    main()