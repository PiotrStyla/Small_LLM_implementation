"""Model SLAYER-Vision-PL: SigLIP (zamrożony) → projector → goLLeM z LoRA."""

import torch
import torch.nn as nn
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import GPT2LMHeadModel, SiglipVisionModel

from slayer_vision.config import IMAGE_TOKENS, MAX_TOTAL_TOKENS


class SlayerVisionModel(nn.Module):
    """Most obraz→język na bazie GoLLeM-110M.

    Inwarianty:
    - encoder obrazu jest zamrożony (jego gradienty nigdy nie powstają),
    - trenowalne są wyłącznie projector MLP i wagi LoRA w LM,
    - tokeny obrazu poprzedzają tokeny tekstu, etykiety pozycji obrazu
      są maskowane (-100), więc LM uczy się wyłącznie generowania tekstu,
    - sekwencja wejściowa (196 tokenów obrazu + tekst) mieści się w oknie
      kontekstu GoLLeM (512).
    """

    def __init__(
        self,
        vision: SiglipVisionModel,
        lm: GPT2LMHeadModel,
        projector_hidden: int = 2048,
        lora_r: int = 16,
        lora_alpha: int = 32,
    ) -> None:
        super().__init__()
        vision_hidden = vision.config.hidden_size
        lm_hidden = lm.config.n_embd

        self.vision = vision
        for param in self.vision.parameters():
            param.requires_grad = False
        self.vision.eval()

        self.projector = nn.Sequential(
            nn.Linear(vision_hidden, projector_hidden),
            nn.GELU(),
            nn.Linear(projector_hidden, lm_hidden),
        )

        lora = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["c_attn"],
            task_type="CAUSAL_LM",
            # GPT-2 używa warstw Conv1D (wagi [in, out]) — LoRA musi wiedzieć,
            # żeby nie transponować macierzy A/B.
            fan_in_fan_out=True,
        )
        # Ewaluacja podaje LM już z wczytanymi adapterami (PeftModel) —
        # wtedy nie zakładamy nowych. Trening podaje surowy GPT2LMHeadModel.
        self.lm: PeftModel = lm if isinstance(lm, PeftModel) else get_peft_model(lm, lora)

        self.image_tokens = IMAGE_TOKENS
        if self.image_tokens + 2 > lm.config.n_positions:
            raise ValueError(
                f"Okno kontekstu LM ({lm.config.n_positions}) nie mieści "
                f"{self.image_tokens} tokenów obrazu i tekstu."
            )

    def trainable_parameters(self):
        return (p for p in self.parameters() if p.requires_grad)

    def forward(
        self,
        pixel_values: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ):
        with torch.no_grad():
            image_hidden = self.vision(pixel_values=pixel_values).last_hidden_state
        patches = image_hidden.shape[1]
        if patches != self.image_tokens:
            raise ValueError(
                f"Encoder zwrócił {patches} tokenów obrazu, "
                f"oczekiwano {self.image_tokens} (patch {self.vision.config.patch_size})."
            )
        if self.image_tokens + input_ids.shape[1] > MAX_TOTAL_TOKENS:
            raise ValueError(
                f"Sekwencja {self.image_tokens + input_ids.shape[1]} tokenów "
                f"przekracza kontekst {MAX_TOTAL_TOKENS}."
            )
        image_embeds = self.projector(image_hidden.to(self.projector[0].weight.dtype))
        text_embeds = self.lm.get_input_embeddings()(input_ids)
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)

        prefix_labels = torch.full(
            (labels.shape[0], image_embeds.shape[1]),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        prefix_mask = torch.ones(
            (attention_mask.shape[0], image_embeds.shape[1]),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )

        return self.lm(
            inputs_embeds=inputs_embeds,
            attention_mask=torch.cat([prefix_mask, attention_mask], dim=1),
            labels=torch.cat([prefix_labels, labels], dim=1),
        )
