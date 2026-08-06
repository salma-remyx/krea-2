"""Generate images with Krea 2 using training-free attention-head pruning.

This is the ``inference.py`` CLI with one extra step: before sampling, the
dispensable-head rule from *Text Template Tokens Are Implicit Semantic Registers
in Diffusion Transformers* (arXiv:2607.19139) is calibrated on a prompt and
applied to the MM-DiT, so a fraction of attention heads is skipped at every
denoising step. It wires ``head_pruning.HeadPruner`` into the existing
``inference._pipeline`` / ``sampling.sample`` path without touching the model
source.

Heads that attend most strongly to the prompt are the ones pruned; pass
``--calibration-prompt`` to score them on a different prompt than the one you
generate with.
"""

import click
import torch

from head_pruning import HeadPruner
from inference import _pipeline, checkpoints
from sampling import prepare, sample


def _calibration_sample(dit, ae, encoder, prompt, width, height, device, dtype):
    """Build one (img, context, t, pos, mask) forward input for scoring heads."""
    patch = dit.config.patch
    align = ae.compression * patch
    height = ((height + align - 1) // align) * align
    width = ((width + align - 1) // align) * align
    txt, txtmask = encoder([prompt])
    noise = torch.randn(
        1,
        ae.channels,
        height // ae.compression,
        width // ae.compression,
        device=device,
        dtype=dtype,
    )
    img, pos, mask = prepare(noise, txt.shape[1], patch, txtmask)
    t = torch.full((1,), 0.5, dtype=dtype, device=device)
    return img, txt, t, pos, mask


@click.command(
    help="Generate images with Krea 2 (K2) using training-free attention-head pruning."
)
@click.argument("prompt")
@click.option("--steps", default=28, show_default=True, help="denoising steps")
@click.option("--cfg", default=4.5, show_default=True, help="CFG scale (0 disables)")
@click.option("--y1", default=0.5, show_default=True, help="timestep-shift mu at min res")
@click.option("--y2", default=1.15, show_default=True, help="timestep-shift mu at max res")
@click.option("--width", default=1024, show_default=True)
@click.option("--height", default=1024, show_default=True)
@click.option("--num-images", default=1, show_default=True)
@click.option("--seed", default=0, show_default=True)
@click.option(
    "--checkpoint",
    envvar="K2_CHECKPOINT",
    default="oss_raw",
    show_default=True,
    type=click.Choice(list(checkpoints)),
)
@click.option("--mu", default=None, type=float, help="timestep-shift mu")
@click.option("--output", default="sample", show_default=True)
@click.option(
    "--prune-fraction",
    default=0.2,
    show_default=True,
    help="fraction of attention heads to prune (paper default ~0.2)",
)
@click.option(
    "--calibration-prompt",
    default=None,
    help="prompt used to score heads (default: the generation prompt)",
)
@click.option(
    "--per-layer",
    is_flag=True,
    help="prune the top fraction within each layer instead of globally",
)
def main(
    prompt,
    steps,
    cfg,
    y1,
    y2,
    width,
    height,
    num_images,
    seed,
    checkpoint,
    output,
    mu,
    prune_fraction,
    calibration_prompt,
    per_layer,
):
    dit, ae, encoder = _pipeline(checkpoint=checkpoint)
    param = next(dit.parameters())
    device, dtype = param.device, param.dtype

    pruner = HeadPruner(fraction=prune_fraction, per_layer=per_layer)
    img, ctx, t, pos, mask = _calibration_sample(
        dit, ae, encoder, calibration_prompt or prompt, width, height, device, dtype
    )
    pruner.calibrate(dit, img, ctx, t, pos, mask)
    click.echo(
        f"pruning {pruner.pruned_fraction:.1%} of attention heads "
        f"({int(pruner.prune_mask.sum())}/{pruner.prune_mask.numel()})"
    )

    pruner.apply(dit)
    try:
        images = sample(
            dit,
            ae,
            encoder,
            [prompt] * num_images,
            width=width,
            height=height,
            steps=steps,
            guidance=cfg,
            seed=seed,
            y1=y1,
            y2=y2,
            mu=mu,
        )
    finally:
        pruner.remove()

    for i, image in enumerate(images):
        out = f"{output}_{i}.png"
        image.save(out)
        click.echo(f"saved {out}")


if __name__ == "__main__":
    main()
