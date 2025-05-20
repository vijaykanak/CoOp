import os.path as osp

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.cuda.amp import GradScaler, autocast

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy, compute_accuracy_i2t
from dassl.utils import load_pretrained_weights, load_checkpoint
from dassl.optim import build_optimizer, build_lr_scheduler

from clip import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

import torch.nn.functional as F
from collections import defaultdict
import pandas as pd

import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

import pandas as pd
import numpy as np

_tokenizer = _Tokenizer()


def load_clip_to_cpu(cfg):
    backbone_name = cfg.MODEL.BACKBONE.NAME
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model


class TextEncoder(nn.Module):
    def __init__(self, clip_model, n_ctx):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = clip_model.dtype
        self.n_ctx = n_ctx

    def forward(self, prompts, tokenized_prompts):
        x = prompts + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        # x = x[torch.arange(x.shape[0]), tokenized_prompts.argmax(dim=-1)] @ self.text_projection

        eos_position = 1 + self.n_ctx  # assuming EOS is always placed right after context
        x = x[:, eos_position, :] @ self.text_projection


        return x


class PromptLearner(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        n_cls = len(classnames)
        n_ctx = cfg.TRAINER.COOP.N_CTX
        ctx_init = cfg.TRAINER.COOP.CTX_INIT
        dtype = clip_model.dtype
        ctx_dim = clip_model.ln_final.weight.shape[0]
        clip_imsize = clip_model.visual.input_resolution
        cfg_imsize = cfg.INPUT.SIZE[0]
        assert cfg_imsize == clip_imsize, f"cfg_imsize ({cfg_imsize}) must equal to clip_imsize ({clip_imsize})"

        if ctx_init:
            # use given words to initialize context vectors
            ctx_init = ctx_init.replace("_", " ")
            n_ctx = len(ctx_init.split(" "))
            prompt = clip.tokenize(ctx_init)
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
            prompt_prefix = ctx_init

        else:
            # random initialization
            if cfg.TRAINER.COOP.CSC:
                print("Initializing class-specific contexts")
                ctx_vectors = torch.empty(n_cls, n_ctx, ctx_dim, dtype=dtype)
            else:
                print("Initializing a generic context")
                ctx_vectors = torch.empty(n_ctx, ctx_dim, dtype=dtype)
            nn.init.normal_(ctx_vectors, std=0.02)
            prompt_prefix = " ".join(["X"] * n_ctx)

        print(f'Initial context: "{prompt_prefix}"')
        print(f"Number of context words (tokens): {n_ctx}")

        self.ctx = nn.Parameter(ctx_vectors)  # to be optimized
        # Learnable weights for each context token
        self.ctx_alpha = nn.Parameter(torch.ones(n_ctx, dtype=dtype), requires_grad=True)  # shape (n_ctx,)
        self.scale_factor=torch.tensor(5.0)


        classnames = [name.replace("_", " ") for name in classnames]
        name_lens = [len(_tokenizer.encode(name)) for name in classnames]
        prompts = [prompt_prefix + " " + name + "." for name in classnames]

        tokenized_prompts = torch.cat([clip.tokenize(p) for p in prompts])
        with torch.no_grad():
            embedding = clip_model.token_embedding(tokenized_prompts).type(dtype)

        # These token vectors will be saved when in save_model(),
        # but they should be ignored in load_model() as we want to use
        # those computed using the current class names
        self.register_buffer("token_prefix", embedding[:, :1, :])  # SOS
        self.register_buffer("token_suffix", embedding[:, 1 + n_ctx :, :])  # CLS, EOS

        self.n_cls = n_cls
        self.n_ctx = n_ctx
        self.tokenized_prompts = tokenized_prompts  # torch.Tensor
        self.name_lens = name_lens
        self.class_token_position = cfg.TRAINER.COOP.CLASS_TOKEN_POSITION

        self.logit_alpha = nn.Parameter(torch.tensor(0.0))   

    def forward(self):
        ctx = self.ctx
        if ctx.dim() == 2:
            ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

        prefix = self.token_prefix
        suffix = self.token_suffix

        # breakpoint()

        if self.class_token_position == "end":
            prompts = torch.cat(
                [
                    prefix,  # (n_cls, 1, dim)
                    ctx,     # (n_cls, n_ctx, dim)
                    suffix,  # (n_cls, *, dim)
                ],
                dim=1,
            )

        elif self.class_token_position == "middle":
            half_n_ctx = self.n_ctx // 2
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i_half1 = ctx[i : i + 1, :half_n_ctx, :]
                ctx_i_half2 = ctx[i : i + 1, half_n_ctx:, :]
                prompt = torch.cat(
                    [
                        prefix_i,     # (1, 1, dim)
                        ctx_i_half1,  # (1, n_ctx//2, dim)
                        class_i,      # (1, name_len, dim)
                        ctx_i_half2,  # (1, n_ctx//2, dim)
                        suffix_i,     # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)

        elif self.class_token_position == "front":
            prompts = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                prefix_i = prefix[i : i + 1, :, :]
                class_i = suffix[i : i + 1, :name_len, :]
                suffix_i = suffix[i : i + 1, name_len:, :]
                ctx_i = ctx[i : i + 1, :, :]
                prompt = torch.cat(
                    [
                        prefix_i,  # (1, 1, dim)
                        class_i,   # (1, name_len, dim)
                        ctx_i,     # (1, n_ctx, dim)
                        suffix_i,  # (1, *, dim)
                    ],
                    dim=1,
                )
                prompts.append(prompt)
            prompts = torch.cat(prompts, dim=0)
        elif self.class_token_position == "learnablePos":
            ctx = self.ctx
            if ctx.dim() == 2:
                ctx = ctx.unsqueeze(0).expand(self.n_cls, -1, -1)

            # Compute per-token softmax weights            
            alpha_softmax = F.softmax(self.ctx_alpha * self.scale_factor, dim=0)  # (n_ctx,)

            avg_class_embeddings = []
            for i in range(self.n_cls):
                name_len = self.name_lens[i]
                class_i = self.token_suffix[i : i + 1, :name_len, :]  # (1, name_len, dim)
                avg_emb = class_i.mean(dim=1)  # (1, dim)
                avg_class_embeddings.append(avg_emb)

                # # Apply weighted avg_emb to context tokens
                # for j in range(self.n_ctx):                    
                #     ctx[i, j, :] = ctx[i, j, :] + alpha[j] * avg_emb

            avg_class_embeddings = torch.cat(avg_class_embeddings, dim=0)  # (n_cls, dim)
            # ctx: (n_cls, n_ctx, dim)
            # alpha: (n_ctx,)
            # avg_class_embeddings: (n_cls, dim)

            # breakpoint()

            # ctx = ctx + alpha.view(1, -1, 1) * avg_class_embeddings.unsqueeze(1)
            # ctx = ctx + avg_class_embeddings.unsqueeze(1)

            dim = avg_class_embeddings.size(1)

            alpha_exp = alpha_softmax.view(1, self.n_ctx, 1).expand(self.n_cls, self.n_ctx, dim)  # (n_cls, n_ctx, dim)
            avg_emb_exp = avg_class_embeddings.unsqueeze(1).expand(self.n_cls, self.n_ctx, dim)  # (n_cls, n_ctx, dim)

            ctx = (1 - alpha_exp) * ctx + alpha_exp * avg_emb_exp  # elementwise (n_cls, n_ctx, dim)   
            
            # ctx = ctx + avg_class_embeddings.unsqueeze(1)  # (n_cls, n_ctx, dim)

            # Extract EOS token safely
            eos_token_id = 49407
            eos_positions = (self.tokenized_prompts == eos_token_id).int().argmax(dim=1)  # (n_cls,)
            batch_indices = torch.arange(self.n_cls, device=eos_positions.device)
            eos_embeddings = self.token_suffix[batch_indices, eos_positions - (1 + self.n_ctx), :].unsqueeze(1)  # (n_cls, 1, dim)

            prompts = torch.cat([self.token_prefix, ctx, eos_embeddings], dim=1)

            # Add padding to ensure prompt is of length 77
            current_len = prompts.shape[1]
            max_len = 77
            pad_len = max_len - current_len

            if pad_len > 0:
                pad_embed = torch.zeros(self.n_cls, pad_len, ctx.shape[-1], dtype=ctx.dtype, device=ctx.device)
                prompts = torch.cat([prompts, pad_embed], dim=1)

            # prompts is now (n_cls, 77, dim)
            # print(prompts.shape)
            # breakpoint()



        else:
            raise ValueError

        return prompts


class CustomCLIP(nn.Module):
    def __init__(self, cfg, classnames, clip_model):
        super().__init__()
        self.prompt_learner = PromptLearner(cfg, classnames, clip_model)
        # breakpoint()
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.image_encoder = clip_model.visual
        self.text_encoder = TextEncoder(clip_model, self.prompt_learner.n_ctx)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype

         

    def forward(self, image, return_features=False):
        image_features = self.image_encoder(image.type(self.dtype))

        prompts = self.prompt_learner()
        tokenized_prompts = self.tokenized_prompts
        text_features = self.text_encoder(prompts, tokenized_prompts)

        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        logit_scale = self.logit_scale.exp()

        if return_features:
            return image_features, text_features, logit_scale

        logits = logit_scale * image_features @ text_features.t()
        return logits



def visualize_batch(image, filtered_logits_per_image, filtered_labels, image_mask):
    # Convert logits to NumPy
    logits_np = filtered_logits_per_image.detach().cpu().numpy()
    labels_np = filtered_labels.cpu().numpy()

    # Create a DataFrame for visualization
    df = pd.DataFrame(logits_np, index=[f"img_{i}_label_{lbl}" for i, lbl in enumerate(labels_np)])
    df.columns = [f"text_{j}" for j in range(df.shape[1])]

    # Optionally round for better readability
    pd.set_option('display.width', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.precision', 3)
    pd.set_option('display.float_format', lambda x: '%.2f' % x)


    print("\nFiltered Logits Per Image (Similarity Matrix):")
    print(df)

    input("\nPress Enter to continue...")


@TRAINER_REGISTRY.register()
class CoOp(TrainerX):
    """Context Optimization (CoOp).

    Learning to Prompt for Vision-Language Models
    https://arxiv.org/abs/2109.01134
    """

    def check_cfg(self, cfg):
        assert cfg.TRAINER.COOP.PREC in ["fp16", "fp32", "amp"]

    def build_model(self):
        cfg = self.cfg
        classnames = self.dm.dataset.classnames

        print(f"Loading CLIP (backbone: {cfg.MODEL.BACKBONE.NAME})")
        clip_model = load_clip_to_cpu(cfg)
        
        if cfg.TRAINER.COOP.PREC == "fp32" or cfg.TRAINER.COOP.PREC == "amp":
            # CLIP's default precision is fp16
            clip_model.float()

        print("Building custom CLIP")
        self.model = CustomCLIP(cfg, classnames, clip_model)

        print("Turning off gradients in both the image and the text encoder")
        for name, param in self.model.named_parameters():
            if "prompt_learner" not in name:
                param.requires_grad_(False)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        # NOTE: only give prompt_learner to the optimizer
        self.optim = build_optimizer(self.model.prompt_learner, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("prompt_learner", self.model.prompt_learner, self.optim, self.sched)

        self.scaler = GradScaler() if cfg.TRAINER.COOP.PREC == "amp" else None

        # Note that multi-gpu training could be slow because CLIP's size is
        # big, which slows down the copy operation in DataParallel
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Multiple GPUs detected (n_gpus={device_count}), use all of them!")
            self.model = nn.DataParallel(self.model)

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        
        prec = self.cfg.TRAINER.COOP.PREC
        if prec == "amp":
            with autocast():
                output = self.model(image)
                # loss = F.cross_entropy(output, label)

                image_features, text_features, logit_scale = self.model(image, return_features=True)

                # Normalize features (already normalized in model, but ensure here if needed)
                image_features = F.normalize(image_features, dim=-1)
                text_features = F.normalize(text_features, dim=-1)

                # Similarity logits: [batch_size, batch_size]
                logits_per_image = logit_scale * image_features @ text_features.t()
                logits_per_text = logits_per_image.t()

                # Ground truth: assume i-th image matches i-th text
                labels = torch.arange(logits_per_image.size(0), device=logits_per_image.device)

                # Symmetric loss (same as used in original CLIP)
                loss_i2t = F.cross_entropy(logits_per_image, labels)
                loss_t2i = F.cross_entropy(logits_per_text, labels)
                loss = (loss_i2t + loss_t2i) / 2
                self.optim.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
        else:
            output = self.model(image)
            # loss = F.cross_entropy(output, label)
            image_features, text_features, logit_scale = self.model(image, return_features=True)

            # Normalize features (already normalized in model, but ensure here if needed)
            image_features = F.normalize(image_features, dim=-1)
            text_features = F.normalize(text_features, dim=-1)

            # Similarity logits: [batch_size, batch_size]
            logits_per_image = logit_scale * image_features @ text_features.t()
            logits_per_text = logits_per_image.t()

            print(logits_per_image.shape)
            print(logits_per_text.shape)
            print(label.shape)

          
            # Text to Image loss
            #############################################################
            # Step 1: Transpose logits to get text-to-image shape
            logits_per_text = logits_per_image.T  # Shape: [37, 32]

            # Step 2: Build text → list of image indices mapping
            text_to_image_indices = defaultdict(list)
            for img_idx, text_idx in enumerate(label):
                text_to_image_indices[int(text_idx)].append(img_idx)

            # Step 3: Create soft target matrix [num_texts, num_images]
            targets_T = torch.zeros_like(logits_per_text)  # [num_texts, num_images]
            for text_idx, image_list in text_to_image_indices.items():
                if len(image_list) > 0:
                    weight = 1.0 / len(image_list)
                    targets_T[text_idx, image_list] = weight

            # Step 4: Filter rows for texts with atleast one matching image
            valid_mask = targets_T.sum(dim=1) > 0  
            filtered_logits_per_text = logits_per_text[valid_mask]
            filtered_targets = targets_T[valid_mask]
            filtered_target_indices = filtered_targets.argmax(dim=1)

            # Step 5: Compute soft cross-entropy using KL divergence
            log_probs = F.log_softmax(filtered_logits_per_text, dim=1)   # log p_model
            loss_t2i = F.kl_div(log_probs, filtered_targets, reduction='batchmean')
            
            #############################################################
            
            
            # Image to text loss
            #############################################################
                   
            
            # Step 1: Build a mask for images whose label indices is in filtered_target_indices
            valid_image_indices = set(filtered_target_indices.tolist())  # Convert to set for faster lookup

            # Now build the mask by checking if the index is in valid_image_indices
            image_mask = torch.tensor([i in valid_image_indices for i in range(len(label))])

            # Step 2: Filter logits and labels
            filtered_logits_per_image = logits_per_image[image_mask]
            filtered_labels = label[image_mask]

            # Step 3: Compute cross entropy loss
            # loss_i2t = F.cross_entropy(filtered_logits_per_image, filtered_labels)
            # loss_i2t = F.cross_entropy(logits_per_image, label)    

            # probs_images = F.softmax(filtered_logits_per_image, dim=1)   # log p_model  
            probs_images = F.softmax(logits_per_image, dim=1)   # log p_model  
            image_no_mask = torch.ones(len(label), dtype=torch.bool)

            log_probs_images = F.log_softmax(filtered_logits_per_image, dim=1)   # log p_model
            num_classes = log_probs_images.size(1)
            filtered_labels_onehot = F.one_hot(filtered_labels, num_classes=num_classes).float()
            loss_i2t = F.kl_div(log_probs_images, filtered_labels_onehot, reduction='batchmean')


            
          

            num_filtered_logits_per_image = filtered_logits_per_image.shape[0]
            num_filtered_logits_per_text = filtered_logits_per_text.shape[0]

            
            print("num_filtered_logits_per_image = ", num_filtered_logits_per_image)
            print("num_filtered_logits_per_text = ", num_filtered_logits_per_text)

        
            
            #############################################################
            print("Epoch Number: ", self.epoch)
            print("logit_scale", logit_scale.item())
            print("Image to text Loss:", loss_i2t.item())
            print("Text to image Loss:", loss_t2i.item())
            print("Num Text-Image pairs: ", filtered_target_indices.shape[0]) 

            logits_i2t=logits_per_image
            logits_t2i=logits_per_image.t()

            # loss_i2t_2 = F.cross_entropy(logits_i2t, label)
            # loss_t2i_2 = F.cross_entropy(logits_t2i, label)
            loss_i2t_2 = loss_i2t
            loss_t2i_2 = loss_t2i
            print("Image to text Loss2:", loss_i2t_2.item())
            print("Text to image Loss2:", loss_t2i_2.item())

            alpha2 = F.softmax(self.model.prompt_learner.ctx_alpha * self.model.prompt_learner.scale_factor, dim=0)

            print("Leanrt pos: ", alpha2)
                 
                



            # loss = loss_i2t
            # loss = loss_t2i
            loss = loss_i2t_2
            # loss = loss_t2i_2
            # loss = (loss_i2t_2 + loss_t2i_2) / 2 
            # loss = (loss_i2t + loss_t2i) / 2 
            # alpha=0.9
            # loss = alpha*loss_i2t + (1-alpha)*loss_t2i 

            # After computing loss_i2t and loss_t2i:
            # alpha = torch.sigmoid(self.model.prompt_learner.logit_alpha)  # constrains alpha in [0, 1]
            # alpha = 0.5 + 0.5 * torch.sigmoid(self.model.prompt_learner.logit_alpha)           

            # detach loss_t2i from the graph for alpha
            # loss = alpha * loss_i2t + (1 - alpha) * loss_t2i.detach()

            # print("alpha: ", alpha.item())     

            # if(self.epoch < 15):
            #     loss = loss_t2i
            # else:
            #     loss = loss_i2t

            # Right after:
            # filtered_logits_per_image = logits_per_image[image_mask]
            # filtered_labels = label[image_mask]

            # visualize_batch(image, filtered_logits_per_image, filtered_labels, image_mask)

            # if(self.epoch == 4):
            #     visualize_batch(image, probs_images, label, image_no_mask)

            



            self.model_backward_and_update(loss)

            # # Step 1: Get model outputs
            # output_batch = self.model(image)  # shape: (batch_size, num_classes)
            # print("output_batch shape: ", output_batch.shape)
            # output_ground_truth_full_batch = self.model(self.ground_truth_images)  # shape: (num_classes, num_classes)
            # print("output_ground_truth_full_batch shape: ", output_ground_truth_full_batch.shape)

            # # Step 2: Select correct rows based on labels
            # # label_image_batch is assumed to be a tensor of shape (batch_size,) with ground truth labels 0, 1, 2, ...
            # label_image_batch = label
            # output_ground_truth_batch = output_ground_truth_full_batch[label_image_batch]  # shape: (batch_size, num_classes)
            # print("output_ground_truth_batch shape: ", output_ground_truth_batch.shape)

            # # Step 3: Expand output_batch to 3D: (batch_size, num_classes, 1)
            # output_batch_expanded = output_batch.unsqueeze(2)  # shape: (batch_size, num_classes, 1)
            # print("output_batch_expanded shape: ", output_batch_expanded.shape)

            # # Step 4: Expand and transpose output_ground_truth_batch to 3D: (batch_size, 1, num_classes)
            # #reference_column = output_ground_truth_batch.unsqueeze(1)  # shape: (batch_size, 1, num_classes)
            # reference_column = output_ground_truth_batch.unsqueeze(1)  # shape: (32, 1, 47)
            # reference_column = reference_column.expand(-1, output_batch.size(1), -1)  # shape: (32, 47, 47)
            # print("reference_column shape: ", reference_column.shape)

            # # Step 5: Concatenate along the last dimension to get shape: (batch_size, num_classes, num_classes + 1)
            # output_enhanced_batch = torch.cat([output_batch_expanded, reference_column], dim=2)  # shape: (batch_size, num_classes, num_classes + 1)
            # print("output_enhanced_batch size: ", output_enhanced_batch.shape)



        # loss_summary = {
        #     "loss": loss.item(),
        #     "acc": compute_accuracy(logits_per_image, label)[0].item(),
        # }

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
        }

        # topk_accuracy = compute_accuracy(output_enhanced_batch, label)
        # print("top_1 accuracy = ", topk_accuracy[0].item())

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()

        return loss_summary

    def parse_batch_train(self, batch):
        input = batch["img"]
        label = batch["label"]
        input = input.to(self.device)
        label = label.to(self.device)
        return input, label

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        names = self.get_model_names()

        # By default, the best model is loaded
        model_file = "model-best.pth.tar"

        if epoch is not None:
            model_file = "model.pth.tar-" + str(epoch)

        for name in names:
            model_path = osp.join(directory, name, model_file)

            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))

            checkpoint = load_checkpoint(model_path)
            state_dict = checkpoint["state_dict"]
            epoch = checkpoint["epoch"]

            # Ignore fixed token vectors
            if "token_prefix" in state_dict:
                del state_dict["token_prefix"]

            if "token_suffix" in state_dict:
                del state_dict["token_suffix"]

            print("Loading weights to {} " 'from "{}" (epoch = {})'.format(name, model_path, epoch))
            # set strict=False
            self._models[name].load_state_dict(state_dict, strict=False)
