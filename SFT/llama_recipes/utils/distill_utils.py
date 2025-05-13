
import torch.nn as nn 
import torch.nn.functional as F

from transformers.models.mllama.configuration_mllama import (
    MllamaConfig,
    MllamaVisionConfig,
)
from transformers.models.mllama.image_processing_mllama import (
    get_all_supported_aspect_ratios
)
from .train_utils import * 


def train(
    student,
    teacher,
    teacher_train_dataloader,
    teacher_eval_dataloader, 
    student_train_dataloader,
    student_eval_dataloader,
    dif_image_processing,
    tokenizer, 
    optimizer, 
    lr_scheduler, 
    gradient_accumulation_steps, 
    train_config, 
    fsdp_config=None, 
    local_rank=None, 
    rank=None, 
    wandb_run=None
):
    # In this case train_config will be of type distill_train_config
    # Create a gradient scaler for fp16
    if train_config.use_fp16 and train_config.enable_fsdp:
        scaler = ShardedGradScaler()
    elif train_config.use_fp16 and not train_config.enable_fsdp:
        scaler = torch.cuda.amp.GradScaler()
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])

    
    autocast = torch.cuda.amp.autocast if train_config.use_fp16 else nullcontext
    train_prep = [] ; train_loss = []
    val_prep = [] ; val_loss =[]

    if train_config.save_metrics:
        if not os.path.exists(train_config.output_dir):
            os.makedirs(train_config.output_dir, exist_ok=True)
        metrics_filename = f"{train_config.output_dir}/metrics_data_{local_rank}-{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"
        train_step_perplexity = [] ; train_step_loss = []
        val_step_loss = [] ; val_step_perplexity = []

    epoch_times = [] ; checkpoint_times = []
    results = {}
    best_val_loss = float("inf")
    total_train_steps = 0
    max_steps_reached = False  # Flag to indicate max training steps reached

    # create projection head and save
    projection_head = ProjectionHead(
        teacher_dim = teacher.config.hidden_size,
        student_dim = student.config.hidden_size, 
    ).train()
    
    # under assumption that model is of type MllamaVisionModel 
    for epoch in range(train_config.num_epochs):
        print(f"Starting epoch {epoch}/{train_config.num_epochs}")
        print(f"train_config.max_train_step: {train_config.max_train_step}")
        # stop when the maximum number of training steps is reached
        if max_steps_reached:
            break
        epoch_start_time = time.perf_counter()
        with MemoryTrace() as memtrace:  # track the memory usage
            student.train()
            total_loss = 0.0
            if len(teacher_train_dataloader) == len(student_train_dataloader):
                total_length = len(teacher_train_dataloader)//gradient_accumulation_steps
            else:
                quit(f"""
                TEACHER dataloader has len: {len(teacher_train_dataloader)}  
                STUDENT dataloader has len: {len(student_train_dataloader)}!
                      """
                    )
            pbar = tqdm(colour="blue", desc=f"Training Epoch: {epoch+1}", total=total_length, dynamic_ncols=True)
            with profile(train_config,local_rank) as profile_context:
                for step, (student_batch,teacher_batch) in enumerate(zip(student_train_dataloader , teacher_train_dataloader)):
                    total_train_steps += 1
                    # stop when the maximum number of training steps is reached
                    if train_config.max_train_step > 0 and total_train_steps > train_config.max_train_step:
                        max_steps_reached = True
                        if not train_config.enable_fsdp or local_rank==0:
                            print("max training steps reached, stopping training, total train steps finished: ", total_train_steps-1)
                        break
                    # since we're only using vision inputs delete all other key-value pairs
                    for batch in [student_batch,teacher_batch]:
                        for key in list(batch.keys()):
                            if key in ["pixel_values" , "aspect_ratio_ids" , "aspect_ratio_mask"]:
                                if train_config.enable_fsdp:
                                    if is_xpu_available():
                                        batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                                    else:
                                        batch[key] = batch[key].to(local_rank)
                                else:
                                    if is_xpu_available():
                                        batch[key] = batch[key].to('xpu:0')
                                    elif torch.cuda.is_available():
                                        batch[key] = batch[key].to('cuda:0')
                                if batch == student_batch:
                                    batch[key] = image_inputs_handler(batch,key,student) if dif_image_processing else batch[key]
                            else:
                                del batch[key]
                    # forward pass
                    with autocast():
                        student_outputs = student(
                            output_attentions=True,
                            output_hidden_states=True,
                            return_dict=True,
                            curious=False, # if False doesn't print out shapes as hidden-states are passed through the model
                            **student_batch
                        )
                        
                        teacher_outputs = teacher(
                            output_attentions=True,
                            output_hidden_states=True,
                            return_dict=True,
                            **teacher_batch
                        )
                        # Collect intermediate features to utilize for distillation. 
                        teacher_intermediates = teacher_outputs[1]
                        student_intermediates = student_outputs[1]

                        # Collect final hidden state. 
                        teacher_final_hidden_state = teacher_outputs[0]
                        student_final_hidden_state = student_outputs[0]
                        '''
                        Use either a projection head or to compute KL divergence between final set of logits. 
                        Interpolate teacher_intermediates to be of same dims as student_intermediates before optionally 
                        passing through a projection head and computing loss, given hidden dims of both teacher & student
                        are the same. 
                        '''
                        # Interpolate so that token count (dim[1]) is the same between teacher and student.
                        teacher_intermediates = interpolate_teacher_states(
                            teacher_intermediates,
                            target_token_count=student_intermediates[0].shape[1] # grab the token count from the first element
                        )
                        # Bilinear interpolation between teacher and student of hidden-states.
                        teacher_final_hidden_state = interpolate_final_hidden_states(
                            teacher_hidden = teacher_final_hidden_state,
                            student_hidden = student_final_hidden_state,
                        )
                        # Project teacher states through a linear layer.
                        # teacher_intermediates = tuple(projection_head(state) for state in teacher_intermediates)
                        
                        # calculate loss between intermediate features using Mean Squared Error
                        loss = 0.0
                        # Iterate through collected intermediate features + final-hidden-states.
                        loss_features = (
                            list(zip(teacher_intermediates, student_intermediates)) + 
                            [(teacher_final_hidden_state,student_final_hidden_state)]
                        )
                        for teacher_state, student_state in loss_features:
                            loss += mse_loss_fn(teacher_state, student_state)
                        # +1 for the final-hidden-state.
                        loss = loss / (len(teacher_intermediates)+1) / gradient_accumulation_steps
                    total_loss += loss.detach().float()
                    if train_config.save_metrics:
                        train_step_loss.append(loss.detach().float().item())
                        train_step_perplexity.append(float(torch.exp(loss.detach().float())))
                    if train_config.use_fp16:
                        # if fp16 is enabled, use gradient scaler to handle gradient update
                        scaler.scale(loss).backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(teacher_train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                scaler.unscale_(optimizer)
                                if train_config.enable_fsdp:
                                    student.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(
                                        student.parameters(),
                                        train_config.gradient_clipping_threshold
                                    )
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            pbar.update(1)
                    else:
                        # regular backpropagation when fp16 is not used
                        loss.backward()
                        if (step + 1) % gradient_accumulation_steps == 0 or step == len(teacher_train_dataloader) - 1:
                            if train_config.gradient_clipping and train_config.gradient_clipping_threshold > 0.0:
                                if train_config.enable_fsdp:
                                    student.clip_grad_norm_(train_config.gradient_clipping_threshold)
                                else:
                                    torch.nn.utils.clip_grad_norm_(
                                        student.parameters(),
                                        train_config.gradient_clipping_threshold
                                    )
                            optimizer.step()
                            optimizer.zero_grad()
                            pbar.update(1)
                    if train_config.use_profiler or train_config.flop_counter:
                        profile_context.step()
                    if train_config.flop_counter and profile_context.is_done():
                        TFlops = profile_context.get_flops_per_sec() / 1e12
                    if wandb_run:
                        if not train_config.enable_fsdp or rank==0:
                            wandb_run.log({
                                'train/epoch': epoch + 1,
                                'train/step': epoch * len(teacher_train_dataloader) + step,
                                'train/loss': loss.detach().float(),
                            })
                    pbar.set_description(
                        f"""Training Epoch: {epoch+1}/{train_config.num_epochs}, step {step}/{len(teacher_train_dataloader)} completed (loss: {loss.detach().float()})"""
                                        )
                    if train_config.save_metrics:
                        save_to_json(
                            metrics_filename,
                            train_step_loss,
                            train_loss,
                            train_step_perplexity,
                            train_prep,
                            val_step_loss,
                            val_loss,
                            val_step_perplexity,
                            val_prep
                        )
                pbar.close()

        epoch_end_time = time.perf_counter()-epoch_start_time
        epoch_times.append(epoch_end_time)
        # Reducing total_loss across all devices if there's more than one CUDA device
        if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        elif torch.cuda.device_count() > 1 and train_config.enable_fsdp:
            dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        train_epoch_loss = total_loss / len(teacher_train_dataloader)
        
        if train_config.enable_fsdp:
            train_epoch_loss = train_epoch_loss/world_size
        train_perplexity = torch.exp(train_epoch_loss)

        train_prep.append(float(train_perplexity))
        train_loss.append(float(train_epoch_loss))

        if not train_config.enable_fsdp or rank==0:
            memtrace.print_stats()

        # Update the learning rate as needed
        lr_scheduler.step()
        should_save_model = train_config.save_model
        '''
        Need to evaluate the sutdent every 'step' against 
        the teacher, if validation was requested. 
        '''
        if train_config.run_validation:
            eval_results = distill_evaluation(
                student,
                teacher,
                train_config,
                student_eval_dataloader,
                teacher_eval_dataloader,
                dif_image_processing,
                local_rank,
                tokenizer,
                wandb_run
            )
            eval_ppl = eval_results[0]
            eval_epoch_loss = eval_results[1]
            temp_val_loss = eval_results[2]
            temp_step_perplexity = eval_results[3]
            if train_config.save_metrics:
                val_step_loss.extend(temp_val_loss)
                val_step_perplexity.extend(temp_step_perplexity)
            should_save_model = train_config.save_model and eval_epoch_loss < best_val_loss
        
        checkpoint_start_time = time.perf_counter()
        '''
        Mix of train_config & student_config as student cfg stores 
        the model name maybe there's a better soln for this...
        '''
        if should_save_model:
            if train_config.enable_fsdp:
                dist.barrier()
            if train_config.use_peft:
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"we are about to save the PEFT modules")
                else:
                    print(f"we are about to save the PEFT modules")
                save_peft_checkpoint(student, train_config.output_dir)
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"PEFT modules are saved in {train_config.output_dir} directory")
                else:
                    print(f"PEFT modules are saved in {train_config.output_dir} directory")
            else:
                if not train_config.enable_fsdp:
                    save_model_checkpoint(student, train_config.output_dir)
                    
                elif fsdp_config.checkpoint_type == StateDictType.FULL_STATE_DICT:
                    print(" Saving the FSDP model checkpoint using FULL_STATE_DICT")
                    print("=====================================================")
                    save_fsdp_model_checkpoint_full(
                        student, optimizer, rank, train_config, epoch=epoch
                    )
                    
                    if train_config.save_optimizer:
                        print(" Saving the FSDP optimizer using FULL_STATE_DICT")
                        print("=====================================================")
                        save_optimizer_checkpoint(
                            student, optimizer, rank, train_config, epoch=epoch
                        )
                    
                elif fsdp_config.checkpoint_type == StateDictType.SHARDED_STATE_DICT:
                    if train_config.save_optimizer:
                        print(" Saving the FSDP model checkpoints using SHARDED_STATE_DICT")
                        print("=====================================================")
                        save_model_and_optimizer_sharded(student, rank, train_config, optim=optimizer)
                    else:
                        print(" Saving the FSDP model checkpoints and optimizer using SHARDED_STATE_DICT")
                        print("=====================================================")
                        save_model_and_optimizer_sharded(student, rank, train_config)
            if train_config.enable_fsdp:
                dist.barrier()
        checkpoint_end_time = time.perf_counter() - checkpoint_start_time
        checkpoint_times.append(checkpoint_end_time)

        if train_config.run_validation:
            if eval_epoch_loss < best_val_loss:
                best_val_loss = eval_epoch_loss
                if train_config.enable_fsdp:
                    if rank==0:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
                else:
                        print(f"best eval loss on epoch {epoch+1} is {best_val_loss}")
            val_loss.append(float(eval_epoch_loss))
            val_prep.append(float(eval_ppl))
        if train_config.enable_fsdp:
            if rank==0:
                print(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
                )
        else:
            print(
                f"Epoch {epoch+1}: train_perplexity={train_perplexity:.4f}, train_epoch_loss={train_epoch_loss:.4f}, epoch time {epoch_end_time}s"
            )

        # Saving the results every epoch to plot later
        if train_config.save_metrics:
            save_to_json(
                metrics_filename,
                train_step_loss,
                train_loss,
                train_step_perplexity,
                train_prep,
                val_step_loss,
                val_loss,
                val_step_perplexity,
                val_prep
            )
    avg_epoch_time = sum(epoch_times)/ len(epoch_times)
    avg_checkpoint_time = sum(checkpoint_times)/ len(checkpoint_times) if len(checkpoint_times) > 0 else 0
    avg_train_prep = sum(train_prep)/len(train_prep)
    avg_train_loss = sum(train_loss)/len(train_loss)
    if train_config.run_validation:
        avg_eval_prep = sum(val_prep)/len(val_prep)
        avg_eval_loss = sum(val_loss)/len(val_loss)
    results['avg_train_prep'] = avg_train_prep
    results['avg_train_loss'] = avg_train_loss
    if train_config.run_validation:
        results['avg_eval_prep'] = avg_eval_prep
        results['avg_eval_loss'] = avg_eval_loss
    results["avg_epoch_time"] = avg_epoch_time
    results["avg_checkpoint_time"] = avg_checkpoint_time
    if train_config.save_metrics:
        results["metrics_filename"] = metrics_filename
    if train_config.flop_counter:
        results["model_tflops"]= TFlops
    #saving the training params including fsdp setting for reference.
    if train_config.enable_fsdp and not train_config.use_peft and rank==0:
        save_train_params(train_config, fsdp_config, rank)

    return results


# Evalutation function
def distill_evaluation(
    student,
    teacher,
    train_config,
    student_eval_dataloader,
    teacher_eval_dataloader,
    dif_image_processing,
    local_rank,
    tokenizer,
    wandb_run
):
    if train_config.enable_fsdp:
        world_size = int(os.environ["WORLD_SIZE"])
    student.eval()
    eval_preds = []
    val_step_loss = []
    val_step_perplexity = []
    eval_loss = 0.0  # Initialize evaluation loss
    total_eval_steps = 0
    with MemoryTrace() as memtrace:
        eval_dataloader = list(zip(student_eval_dataloader,teacher_eval_dataloader))
        for step, (student_batch,teacher_batch) in enumerate(
            tqdm(
                eval_dataloader, colour="green", desc="evaluating Epoch", dynamic_ncols=True
            )
        ):
            total_eval_steps += 1
            # stop when the maximum number of eval steps is reached
            if (
                train_config.max_eval_step > 0 and
                total_eval_steps > train_config.max_eval_step
            ):
                if not train_config.enable_fsdp or local_rank==0:
                    print(
                        "max eval steps reached, stopping evaluation, total_eval_steps: ",
                        total_eval_steps - 1
                         )
                break
            for batch in [student_batch,teacher_batch]:
                for key in list(batch.keys()):
                    if key in ["pixel_values" , "aspect_ratio_ids" , "aspect_ratio_mask"]:
                        if train_config.enable_fsdp:
                            if is_xpu_available():
                                batch[key] = batch[key].to(torch.device(f"xpu:{local_rank}"))
                            else:
                                batch[key] = batch[key].to(local_rank)
                        else:
                            if is_xpu_available():
                                batch[key] = batch[key].to('xpu:0')
                            elif torch.cuda.is_available():
                                batch[key] = batch[key].to('cuda:0')
                        if batch == student_batch:
                            batch[key] = image_inputs_handler(batch,key,student) if dif_image_processing else batch[key]
                    else:
                        del batch[key]
            # Ensure no gradients are computed for this scope to save memory
            with torch.no_grad():
                student_outputs = student(
                            output_attentions=True,
                            output_hidden_states=True,
                            return_dict=True,
                            curious=False, # if False doesn't print out shapes as hidden-states are passed through the model
                            **student_batch
                        )
                teacher_outputs = teacher(
                            output_attentions=True,
                            output_hidden_states=True,
                            return_dict=True,
                            **teacher_batch
                        )
                # Collect intermediate features to utilize for distillation. 
                teacher_intermediates = teacher_outputs[1]
                student_intermediates = student_outputs[1]

                # Collect final hidden state. 
                teacher_final_hidden_state = teacher_outputs[0]
                student_final_hidden_state = student_outputs[0]
                
                # Interpolate so that token count (dim[1]) is the same between teacher and student.
                teacher_intermediates = interpolate_teacher_states(
                        teacher_intermediates,
                        target_token_count=student_intermediates[0].shape[1] # grab the token count from the first element
                    )
                # Bilinear interpolation between teacher and student of hidden-states.
                teacher_final_hidden_state = interpolate_final_hidden_states(
                        teacher_hidden = teacher_final_hidden_state,
                        student_hidden = student_final_hidden_state,
                )
                # Project teacher states through a linear layer.
                # teacher_intermediates = tuple(projection_head(state) for state in teacher_intermediates)
                
                # calculate loss between intermediate features using Mean Squared Error
                loss = 0.0
                # Iterate through collected intermediate features + final-hidden-states.
                loss_features = (
                    list(zip(teacher_intermediates, student_intermediates)) + 
                    [(teacher_final_hidden_state,student_final_hidden_state)]
                )
                for teacher_state, student_state in loss_features:
                    loss += mse_loss_fn(teacher_state, student_state)
                # Normalize loss; +1 for the final-hidden-state.
                loss = loss / (len(teacher_intermediates)+1)
                if train_config.save_metrics:
                    val_step_loss.append(loss.detach().float().item())
                    val_step_perplexity.append(float(torch.exp(loss.detach().float())))
                eval_loss += loss.detach().float()
            '''
            Suppose to grab logits from outputs, but since we're using just the vision 
            models right now no logits are returned, but for the evaluation models 
            we may want to figure out a way to incorporate text heads for better evaluation. 
            '''
    # If there's more than one CUDA device, reduce evaluation loss across all devices
    if is_xpu_available() and (torch.xpu.device_count() > 1 and train_config.enable_fsdp):
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
    if torch.cuda.device_count() > 1 and train_config.enable_fsdp:
        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)

    # Compute average loss and perplexity
    eval_epoch_loss = eval_loss / len(eval_dataloader)
    if train_config.enable_fsdp:
        eval_epoch_loss = eval_epoch_loss/world_size
    eval_ppl = torch.exp(eval_epoch_loss)

    # Print evaluation metrics
    if train_config.enable_fsdp:
        if local_rank==0:
            print(f" {eval_ppl=} {eval_epoch_loss=}")
    else:
        print(f" {eval_ppl=} {eval_epoch_loss=}")

    if wandb_run:
        wandb_run.log(
            {
                'eval/perplexity': eval_ppl,
                'eval/loss': eval_epoch_loss,
            }, 
            commit=False
        )

    return eval_ppl, eval_epoch_loss, val_step_loss, val_step_perplexity


def get_student_aspect_ratio_id(
    aspect_ratio_id,
    aspect_ratio_mask,
    student_cfg:MllamaVisionConfig
):
    """
    Backtracking logic to ensure the correct aspect_ratio_id
    is fetched in the due to differences between student final 
    cfg and warm-up resolutions. 
    """
    _,_,processor_tile_count = aspect_ratio_mask.shape
    processor_aspect_ratios = [list(i) for i in get_all_supported_aspect_ratios(processor_tile_count)]
    
    batch_size,_ = aspect_ratio_id.shape
    if batch_size > 1:
        student_aspect_ratio_ids = []
        for idx in torch.unbind(aspect_ratio_id,dim=0):
            aspect_ratio = processor_aspect_ratios[idx]
            student_aspect_ratio_ids.append(
                torch.tensor([[student_cfg.supported_aspect_ratios.index(aspect_ratio)]])
            )
        return torch.stack(student_aspect_ratio_ids, dim=0)
    else:
        aspect_ratio = processor_aspect_ratios[aspect_ratio_id]
        try:
            return torch.tensor([[student_cfg.supported_aspect_ratios.index(aspect_ratio)]])
        except Exception as E:
            quit(
        f"""
        Error occured check the aspect ratio configuration found {aspect_ratio} must not be avaliable in these list of options:
        {student_cfg.supported_aspect_ratios}!
        """
            )


def pad_aspect_ratio_mask(aspect_ratio_mask,cfg:MllamaVisionConfig):
    """
    Pad the aspect_ratio_mask to the required student length.
    Allow the processor to determine which tiles should be used
    for the forward pass and then pad the additional tiles for the 
    student to be 0. 
    """
    batch_size,_,processor_tile_count = aspect_ratio_mask.shape
    expected_tile_count = cfg.max_num_tiles
    extra_zeros = expected_tile_count - processor_tile_count
    if batch_size > 1:
        aspect_ratio_masks=[]
        for indv_mask in torch.unbind(aspect_ratio_mask,dim=0):
            zeros_to_add = torch.zeros( 
                (1, extra_zeros),
                dtype=aspect_ratio_mask.dtype
            ).to(aspect_ratio_mask.device)
            aspect_ratio_masks.append(
                torch.cat([indv_mask, zeros_to_add], dim=-1)
            )
        return torch.stack(aspect_ratio_masks, dim=0)
    else:
        if (
            expected_tile_count != processor_tile_count and 
            expected_tile_count > processor_tile_count
        ):
            zeros_to_add = torch.zeros( 
                (1, 1, extra_zeros),
                dtype=aspect_ratio_mask.dtype
            ).to(aspect_ratio_mask.device)
            return torch.cat(
                [aspect_ratio_mask, zeros_to_add], dim=-1
            )
        elif expected_tile_count < processor_tile_count:
            raise ValueError(
                f"Model tile count: {expected_tile_count} shouldn't be less than processor tile count {processor_tile_count}!"
            )
        else:
            return aspect_ratio_mask


def interpolate_pixel_values(
    pixel_values: torch.Tensor,
    student_cfg: MllamaVisionConfig,
):
    """
    Interpolates a 6D tensor while preserving all dimensions but resizing:
      - Number of tiles per image
      - Tile height & width
    Args:
        pixel_values (torch.Tensor): Tensor of shape (B, I, T, C, H, W).
        target_tile_size (int): New number of tiles per image (T' dimension).
        target_img_size (tuple): New (Height, Width) for each tile.
    Returns:
        torch.Tensor: Interpolated tensor with shape (B, I, T', C, new_H, new_W).
    """
    target_tile_size = student_cfg.max_num_tiles
    target_img_size = (student_cfg.image_size , student_cfg.image_size)
    
    B, I, T, C, H, W = pixel_values.shape  # Original shape

    # Step 1: Reshape (Merge batch and images for processing)
    pixel_values_reshaped = pixel_values.view(B * I, T, C, H, W)  # Shape: [B*I, T, C, H, W]

    # Step 2: Interpolate Tile Count (Temporal dimension) using 3D interpolation
    pixel_values_resized_tiles = itp(
        pixel_values_reshaped.permute(0, 2, 1, 3, 4),  # Move Tile (T) to Channels dim
        size=(target_tile_size, H, W),
        mode="trilinear",  # Smooth resizing across tiles
        align_corners=False
    ).permute(0, 2, 1, 3, 4)  # Move back to original dim order

    # Step 3: Interpolate Spatial Dimensions (H, W)
    pixel_values_final = itp(
        pixel_values_resized_tiles.reshape(B * I * target_tile_size, C, H, W),  # Flatten for 2D interpolation
        size=target_img_size,  # Resize each tile spatially
        mode="bilinear",
        align_corners=False
    ).view(B, I, target_tile_size, C, target_img_size[0], target_img_size[1])  # Reshape back to 6D

    return pixel_values_final


def image_inputs_handler(
    batch, 
    key:str,
    student,
):
    """
    Takes in any accepted key after image processing and performs 
    the nescary interpolation to ensure homogeniety in inputs 
    before passing into the final vision model. The final goal 
    is to be able to gradually increase image resolution and tile 
    count for the student's embedding layers.
    """
    if key == "pixel_values":
        batch[key] = interpolate_pixel_values(
            batch[key], 
            student.config,
        )
    elif key == "aspect_ratio_ids": 
        batch[key] = get_student_aspect_ratio_id(
            batch[key],
            batch["aspect_ratio_mask"],
            student.config,
        )
    elif key ==  "aspect_ratio_mask":
        batch[key] = pad_aspect_ratio_mask(
            batch[key], 
            student.config,
        )
    else:
        raise KeyError(f"{key} is not a key option!")
    
    return batch[key]

    
def mse_loss_fn(
    prediction: torch.Tensor,
    target: torch.Tensor
) -> torch.Tensor:
    """
    Computes the Mean Squared Error (MSE) loss between the prediction and target tensors.
    Args:
        prediction (torch.Tensor): The predicted output tensor.
        target (torch.Tensor): The ground truth tensor.
    Returns:
        torch.Tensor: The computed MSE loss, a SCALAR tensor. 
    """
    return nn.MSELoss()(prediction, target)


# USE 1D interpolation to align the number of tokens 
def interpolate_teacher_states(teacher_states, target_token_count):
    """
    Interpolates a tuple of teacher intermediate hidden states to match a target token count.
    Args:
        teacher_states (tuple[torch.Tensor]): A tuple of tensors, each of shape [B, T_teacher, D].
        target_token_count (int): The target number of tokens (T_target) to interpolate to.
    Returns:
        tuple[torch.Tensor]: A tuple of tensors, each of shape [B, T_target, D].
    """
    interpolated_states = []
    for state in teacher_states:
        # state has shape [B, T_teacher, D]
        B, T_teacher, D = state.shape
        # Transpose to shape [B, D, T_teacher] so that the token dimension is last.
        state_transposed = state.transpose(1, 2)  # [B, D, T_teacher]
        # Interpolate along the token dimension to target_token_count
        # For 1D data, mode 'linear' is appropriate.
        interpolated = F.interpolate(state_transposed, size=target_token_count, mode='linear', align_corners=False)
        # Transpose back to shape [B, T_target, D]
        interpolated = interpolated.transpose(1, 2)
        interpolated_states.append(interpolated)
    return tuple(interpolated_states)


'''
Use bilinear interpolation to map teacher-hidden states to student.
For some reason tying the function to a variable presents getting thrown 
an attribute error saying that F is of type 'int' and therefore has no 
attribute interpolate, WTF. 
'''
itp = F.interpolate 
def interpolate_final_hidden_states(
    teacher_hidden: torch.Tensor,
    student_hidden: torch.Tensor
) -> torch.Tensor:
    """
    Interpolates the teacher's final hidden states to match the student's dimensions 
    using bilinear interpolation.
    Args:
        teacher_hidden (torch.Tensor): Teacher's hidden states of shape [1, 1, 4, 1601, 7680]
        student_hidden (torch.Tensor): Student's hidden states of shape [1, 1, 25, 257, 7680]
    Returns:
        torch.Tensor: Interpolated teacher hidden states matching student shape (5D).
    """
    # Extract target sequence and token count from student
    target_seq_len = student_hidden.shape[2]  # 25
    target_token_count = student_hidden.shape[3]  # 257
    # Reshape to 4D (collapse batch and feature dims)
    B, C, S_T, D_T, F = teacher_hidden.shape  # Extract teacher dimensions
    teacher_reshaped = teacher_hidden.view(B * C, S_T, D_T, F).permute(0, 3, 1, 2)  # Shape: [B*C, F, S_T, D_T]
    # Perform bilinear interpolation on (sequence length, token count)
    teacher_interpolated = itp(
        teacher_reshaped, 
        size=(target_seq_len, target_token_count),  # Resize teacher to match student
        mode="bilinear",
        align_corners=False
    )
    # Reshape back to 5D
    teacher_interpolated = teacher_interpolated.permute(0, 2, 3, 1).view(B, C, target_seq_len, target_token_count, F)
    
    return teacher_interpolated


class ProjectionHead(nn.Module):
    def __init__(self, teacher_dim, student_dim, hidden_dim=None):
        super(ProjectionHead, self).__init__()
        # If hidden_dim is provided, use a small MLP; otherwise, use a single linear layer.
        if hidden_dim is None:
            self.projection = nn.Linear(teacher_dim, student_dim)
        else:
            self.projection = nn.Sequential(
                nn.Linear(teacher_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, student_dim)
            )
    def forward(self, x):
        return self.projection(x)





#
