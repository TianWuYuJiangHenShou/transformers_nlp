#python ./further_pre_training/create_pretraining_data.py \
#  --input_file=data/corpus.txt \
#  --output_file=data/corpus.tfrecord \
#  --vocab_file=/root/workspace/berts/chinese_L-12_H-768_A-12/vocab.txt \
#  --do_lower_case=True \
#  --max_seq_length=128 \
#  --max_predictions_per_seq=20 \
#  --masked_lm_prob=0.15 \
#  --random_seed=12345 \
#  --dupe_factor=5

#rm pretrain.log
#python ./further_pre_training/run_pretraining.py \
#  --input_file=data/corpus.tfrecord \
#  --output_dir=state_dict/corpus_pretrain_gpu \
#  --do_train=True \
#  --do_eval=True \
#  --bert_config_file=/root/workspace/berts/chinese_L-12_H-768_A-12/bert_config.json \
#  --init_checkpoint=/root/workspace/berts/chinese_L-12_H-768_A-12/bert_model.ckpt \
#  --train_batch_size=32 \
#  --max_seq_length=128 \
#  --max_predictions_per_seq=20 \
#  --num_train_steps=100000 \
#  --num_warmup_steps=10000 \
#  --save_checkpoints_steps=10000 \
#  --learning_rate=5e-5
#  --use_tpu=True


python ./fine_tune/convert_tf_checkpoint_to_pytorch.py \
  --tf_checkpoint_path ./state_dict/corpus_pretrain/model.ckpt-10000 \
  --bert_config_file ./state_dict/corpus_pretrain/bert_config.json \
  --pytorch_dump_path ./state_dict/corpus_pretrain/pytorch_model.bin
