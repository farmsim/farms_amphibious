"""Plot network"""

import os

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import animation

from PIL import Image
from moviepy.editor import VideoClip

from farms_core import pylog
from farms_core.utils.profile import profile
from farms_core.analysis.plot import plt_style_options

from farms_amphibious.data.data import AmphibiousData
from farms_amphibious.model.options import AmphibiousOptions
from farms_amphibious.utils.network import plot_networks_maps
from farms_amphibious.utils.parse_args import parser_postprocessing


def parse_args():
    """Parse args"""
    parser = parser_postprocessing(description='Plot amphibious network')
    parser.add_argument(
        '--single_frame',
        action='store_true',
        default=False,
        help='Render single frame',
    )
    parser.add_argument(
        '--no_text',
        action='store_true',
        default=False,
        help='No text',
    )
    return parser.parse_args()


def main(use_moviepy=True):
    """Main"""

    # Clargs
    args = parse_args()
    _, extension = os.path.splitext(args.output)

    # Style
    matplotlib.use('Agg')
    plt_style_options()

    # Setup
    animat_options = AmphibiousOptions.load(args.animat)
    # simulation_options = SimulationOptions.load(args.simulation)
    data = AmphibiousData.from_file(args.data)
    network_anim = plot_networks_maps(
        data=data,
        animat_options=animat_options,
        show_text=not args.no_text,
        animation_only=True,
        show_all=False,
        figsize=(14, 7),
    )[0]
    fig = plt.gcf()
    cax = plt.gca()
    fig.tight_layout()
    fig.set_size_inches(14, 7)

    # Draw to save background
    fig.canvas.draw()
    background = fig.canvas.copy_from_bbox(fig.bbox)

    # Animation elements
    elements = network_anim.animation_elements()

    # Render
    if extension == '.png':
        # Save every frame
        for frame in range(1 if args.single_frame else network_anim.n_frames):
            pylog.debug('Saving frame %s', frame)
            # Restore
            fig.canvas.restore_region(background)
            # Update
            network_anim.animation_update(frame)
            # Draw
            for element in elements:
                if element is not None:
                    cax.draw_artist(element)
            # Save
            img = Image.frombytes(
                'RGB',
                fig.canvas.get_width_height(),
                fig.canvas.tostring_rgb(),
            )
            os.makedirs(os.path.dirname(args.output), exist_ok=True)
            img.save(args.output.format(frame=frame))
    elif extension == '.mp4':
        if use_moviepy:
            # Use Moviepy
            fps = 1/(1e-3*network_anim.interval)
            duration = network_anim.timestep*network_anim.n_iterations
            n_frames = network_anim.n_frames

            def make_frame(current_time):
                """Make frame"""
                frame = min([int(current_time*fps), network_anim.n_frames])
                pylog.debug('Saving frame %s/%s', frame, n_frames)
                # Restore
                fig.canvas.restore_region(background)
                # Update
                network_anim.animation_update(frame)
                # Draw
                for element in elements:
                    if element is not None:
                        cax.draw_artist(element)
                return np.array(fig.canvas.renderer.buffer_rgba())[:, :, :3]

            anim = VideoClip(make_frame, duration=duration)
            assert fps is not None
            anim.write_videofile(
                filename=args.output,
                fps=fps,
                codec='libx264',
                logger='bar',
            )
            anim.close()
        else:
            # Use Matplotlib
            moviewriter = animation.writers['ffmpeg'](
                fps=1/(1e-3*network_anim.interval),
                metadata=dict(
                    title='FARMS network',
                    artist='FARMS',
                    comment='FARMS network',
                ),
                extra_args=['-vcodec', 'libx264'],
            )
            with moviewriter.saving(
                    fig=fig,
                    outfile=args.output,
                    dpi=100,
            ):
                for frame in range(network_anim.n_frames):
                    pylog.debug('Saving frame %s', frame)
                    network_anim.animation_update(frame)
                    moviewriter.grab_frame()
    else:
        raise Exception(f'Unknown file extension {extension}')
    pylog.info('Saved to %s', args.output)


if __name__ == '__main__':
    profile(main)
    # main()
