"""Fast inference entry point for Krea 2 with dual feature caching.

Runs the same pipeline as ``inference.py`` (reusing its model/autoencoder/
encoder build and ``sampling.sample``) but wraps the MM-DiT in
:class:`feature_cache.DualCacheDiT` so that transformer-block outputs are
cached across denoising steps and reused to skip computation on later steps.

Pass ``--cache-static 0 --cache-dynamic 0`` for exact parity with
``inference.py``.
"""

import click

from feature_cache import DualCacheDiT
from inference import _pipeline, checkpoints
from sampling import sample


@click.command(help="Generate images with Krea 2 using dual feature caching.")
@click.argument("prompt")
@click.option(
    "--steps", default=28, show_default=True, help="number of denoising steps"
)
@click.option(
    "--cfg",
    default=4.5,
    show_default=True,
    help="classifier-free guidance scale (0 disables CFG)",
)
@click.option(
    "--y1", default=0.5, show_default=True, help="timestep-shift mu at min resolution"
)
@click.option(
    "--y2", default=1.15, show_default=True, help="timestep-shift mu at max resolution"
)
@click.option("--width", default=1024, show_default=True)
@click.option("--height", default=1024, show_default=True)
@click.option(
    "--num-images",
    default=1,
    show_default=True,
    help="number of images to generate from the prompt",
)
@click.option(
    "--seed", default=0, show_default=True, help="base seed; image i uses seed + i"
)
@click.option(
    "--checkpoint",
    envvar="K2_CHECKPOINT",
    default="oss_raw",
    show_default=True,
    type=click.Choice(list(checkpoints)),
)
@click.option(
    "--mu",
    default=None,
    help="timestep-shift mu",
    type=float,
)
@click.option(
    "--output", default="sample", show_default=True, help="output filename prefix"
)
@click.option(
    "--cache-static",
    default=4,
    show_default=True,
    type=int,
    help="first N transformer blocks whose output is frozen after step 0",
)
@click.option(
    "--cache-dynamic",
    default=4,
    show_default=True,
    type=int,
    help="next N transformer blocks whose output is refreshed every other step",
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
    cache_static,
    cache_dynamic,
):
    dit, ae, encoder = _pipeline(checkpoint=checkpoint)
    if cache_static or cache_dynamic:
        dit = DualCacheDiT(dit, n_static=cache_static, n_dynamic=cache_dynamic)

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
    for i, image in enumerate(images):
        out = f"{output}_{i}.png"
        image.save(out)
        click.echo(f"saved {out}")


if __name__ == "__main__":
    main()
