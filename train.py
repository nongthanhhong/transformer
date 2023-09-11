from gpu_train import *
from tpu_train import *
from config import *
import os
import argparse

def train_w_gpu(train_data_loader, 
                    val_data_loader, 
                    input_vocab_size,
                    output_vocab_size):
    # Create the model
    model = Transformer(max_len=max_len_input,
                        input_vocab_size=input_vocab_size,
                        output_vocab_size=output_vocab_size,
                        num_layers=num_layers,   
                        heads=num_heads, 
                        d_model=d_model, 
                        d_ff=d_ff, 
                        dropout=drop_out_rate, 
                        bias=True)
    
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(model)
    
    model.to(device)

    # Define your optimizer
    optimizer = create_optimizer(model.parameters())

    #load if checkpoint
    trained_epoch = 0
    
    if os.path.isfile(saved_checkpoint_path):
        print(f"Load model from check point: {saved_checkpoint_path}")
        trained_epoch, model, optimizer = load_checkpoint(model, optimizer, saved_checkpoint_path)

    # Create a SummaryWriter instance for TensorBoard logging
    writer = SummaryWriter()

    # setup warmup schedule and optimizer
    # Define total number of training steps
    total_steps = len(train_data_loader)
    # Define warmup proportion
    warmup_proportion = 0.1
    # Compute the number of warmup steps
    warmup_steps = int(total_steps * warmup_proportion)
    # Define your scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min((step + 1) / warmup_steps, 1))

    # Train the model
    print("\n=============== Start Training Phase ===============")
    start_time = time.time()
    best_epoch = 0
    pre_loss = 10**9

    for epoch in range(trained_epoch + 1, trained_epoch + num_epochs + 1):
        print(f'EPOCH {epoch}:')
        train_loss = train(train_data_loader, 
                           model,
                           loss_fn,
                           optimizer,
                           scheduler,
                           writer,
                           log_interval)
        val_loss, avg_bleu = evaluate(val_data_loader,
                            model,
                            loss_fn)
        log_progress(epoch, train_loss, val_loss, avg_bleu)
        writer.add_scalar("Loss/train_epoch", train_loss, epoch, val_loss)
        
        #check best epoch
        if val_loss < pre_loss:
            pre_loss = val_loss
            best_epoch = epoch
            
            now = datetime.datetime.now()
            checkpoint_path = f"{ckpt_dir}/transformer_{optimizer_name}_epoch_{epoch}_loss_{val_loss:.4f}_BLEU_{avg_bleu}_m{now.month}_d{now.day}_{now.hour}h_{now.minute}m.pt"
            save_checkpoint(model, optimizer, epoch, val_loss, checkpoint_path)

        
    end_time = time.time() - start_time
    print(f'Training take: {round(end_time, 3)} seconds ~ {round(end_time/60, 3)} minutes ~ {round(end_time/3600, 3)} hours')
    print(f'Best Epoch: {best_epoch} with val_loss = {round(pre_loss, 8)}')


def train_w_tpu():
    assert os.environ['COLAB_TPU_ADDR'], 'Make sure to select TPU from Edit > Notebook settings > Hardware accelerator'
    

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    
    # Set the file paths and other parameters
    train_input = "dataset/PhoMT/tokenization/train/train.en"
    train_output = "dataset/PhoMT/tokenization/train/train.vi"
    val_input = "dataset/PhoMT/tokenization/dev/dev.en"
    val_output = "dataset/PhoMT/tokenization/dev/dev.vi"

    parser.add_argument('--train-input', type=str, help="path to train input file for train model",
                        default=train_input)
    parser.add_argument('--train-output', type=str, help="path to train output file for train model",
                        default=train_output)
    parser.add_argument('--val-input', type=str, help="path to validation input file for train model",
                        default=val_input)
    parser.add_argument('--val-output', type=str, help="path to validation ouput file for train model",
                        default=val_output)

    args = parser.parse_args()

    
    

    train_data_loader, val_data_loader, input_tokenizer, output_tokenizer = Data(args.train_input, args.train_output, 
                                                                                 args.val_input, args.val_output, 
                                                                                 batch_size)
    input_vocab_size = input_tokenizer.vocab_size()
    output_vocab_size = output_tokenizer.vocab_size()

    print(f'Size of English vocab: {input_vocab_size}\nSize of Vietnamese vocab: {output_vocab_size}')

    if train_device == 'gpu':
        train_w_gpu(train_data_loader, 
                    val_data_loader,
                    input_vocab_size,
                    output_vocab_size)
    if train_device == 'tpu':
        train_w_tpu()