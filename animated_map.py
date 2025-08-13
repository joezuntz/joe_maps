from .maps import Map
from . import anim
import matplotlib.patches
from .styles import mmc_colors, PC
import matplotlib.pyplot as plt
import numpy as np
from . import zorders
from .figure import DuckTransform


class AnimatedMap(Map):
    def __init__(self, *args, delta, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.timeline = anim.Timeline(self.ax)
        self.init_rect = matplotlib.patches.Rectangle((2, 28), 0.01, 0.01, alpha=0, transform=PC)
        self.ax.add_patch(self.init_rect)
        self.current_time = 0.0
        self._current_border_color = "#000000FF"
        self._current_border_width = 1.0
        self._current_extent = self.ax.get_extent(crs=PC)
        self._current_label_offsets = {}
        self.fog_img = None
        self.verbose = True

    def log(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)

    def add_label(
        self,
        place,
        offset=(0, 0),
        facecolor=mmc_colors.wheat,
        textcolor=mmc_colors.dark_blue,
        edgecolor=mmc_colors.dark_blue,
        zorder=zorders.LABELS,
        **kwargs,
    ):
        if isinstance(place, str):
            y, x = self.locations[place]
        else:
            y, x = place
        box = dict(boxstyle="round", facecolor=facecolor, alpha=1, edgecolor=edgecolor)
        label = self.ax.text(
            x + offset[0],
            y + offset[1],
            place,
            fontsize=14,
            bbox=box,
            fontname="Arial",
            color=textcolor,
            zorder=zorder,
            transform=PC,
        )
        self.labels[place] = label
        self._current_label_offsets[place] = offset
        label.set_visible(False)
        self.log(f"Showing label {place} at", self.current_time)
        self.show(label)
        return label
    
    def add_text(self, place, text, time=None, offset=(0, 0), facecolor=mmc_colors.wheat, textcolor=mmc_colors.dark_blue, edgecolor=mmc_colors.dark_blue, bbox=True, boxstyle="round", **kwargs):
        text = super().add_text(place, text, offset=offset, facecolor=facecolor, textcolor=textcolor, edgecolor=edgecolor, bbox=bbox, boxstyle=boxstyle, **kwargs)
        text.set_visible(False)
        self.log(f"Showing text '{text}' at {place} at", self.current_time)
        self.show(text)
        if time is not None:
            frame = (self.current_time + time) / self.delta
            self.log(f"Hiding text '{text}' at {place} at", self.current_time + time)
            self.timeline.add_transition(anim.make_disappear, frame, text)
            self.current_time += time
        return text
    

    def add_point(self, place, *args, **kwargs):
        points = super().add_point(place, *args, **kwargs)
        points.set_visible(False)
        self.log(f"Showing point at {place} at", self.current_time)
        self.show(points)
        return points

    def _animate_journey(self, journey, time, speed, fade_after=0, fade_to=0.0):
        if time is None and speed is None:
            raise ValueError("Set time or speed for animated map journeys")
        elif time is None:
            time = journey.length() / speed

        # This hides the journey to start with
        journey.hide()

        # Work out timing parameters
        end_time = self.current_time + time
        start_frame = self.current_time / self.delta
        end_frame = end_time / self.delta
        self.log(
            f"Journey will take from time {self.current_time:.2f} to {end_time:.2f} == frame {start_frame:.1f} to {end_frame:.1f}"
        )

        if self.fog is not None and journey.fog_clearance is not None:
            supe = super()
            def unmask_journey(f):
                return supe.unmask_journey(journey, frac=f)
            self.timeline.add_fraction_updater(unmask_journey, start_frame, end_frame)

        self.timeline.add_fraction_updater(journey.animate, start_frame, end_frame)

        # Add the animation to the timeline
        # self.timeline.add_fraction_updater(anim.wipe_journey, start_frame, end_frame, self.ax, direction, journey)
        self.current_time = end_time

    
        if fade_after > 0:
            fade_frame = (self.current_time + fade_after) / self.delta
            self.log(f"Journey will fade out from end to  {fade_frame:.1f}")
            self.timeline.add_fraction_updater(
                journey.fade_out, end_frame, fade_frame, fade_to
            )

    def add_journey(
        self, *args, time=None, speed=None, fade_after=0, fade_to=0, **kwargs
    ):
        journey = super().add_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, fade_after=fade_after, fade_to=fade_to)
        return journey

    def add_return_journey(
        self, journey, time=None, speed=None, fade_after=0, fade_to=0,
    ):
        return_journey = journey.get_reverse()
        return_journey.draw(self.ax)
        self._animate_journey(return_journey, time, speed, fade_after=fade_after, fade_to=fade_to)
        self.journeys.append(return_journey)
        return return_journey

    def add_gpx_journey(
        self, *args, speed=None, time=None, fade_after=0, fade_to=0, **kwargs
    ):
        journey = super().add_gpx_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, fade_after=fade_after, fade_to=fade_to)
        return journey
    
    def add_band_journey(
        self, *args, time=None, speed=None,fade_after=0, fade_to=0, **kwargs):
        journey = super().add_band_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, fade_after=fade_after, fade_to=fade_to)
        return journey

    def save(self, path, **kwargs):
        interval = self.delta * 1000
        frames = int(np.ceil(self.current_time / self.delta)) + 1
        if self.fig.post_transform is not None:
            if "writer" in kwargs and (kwargs["writer"] not in ["None", "ffmpeg_file"]):
                raise ValueError("Writer must be 'None' or 'ffmpeg_file' for AnimatedMap if transforms are used")
            kwargs.pop("writer", None)
        self.log(f"\nAnimation will have {frames} frames\n")
        self.timeline.save(self.fig, path, interval=interval, frames=frames, writer="ffmpeg_file", **kwargs)

    def delay_last_transition(self):
        self.timeline.transitions[-1][1] = self.current_time / self.delta

    def set_border_width(self, width, time=None):
        supe = super()

        # If this is an instant change we add a transition function
        # at the current time
        if time is None:
            def f():
                return supe.set_border_width(width)
            self.timeline.add_transition(f, self.current_time / self.delta)
            self._current_border_width = width
            self.log("Setting border width to", width, "at frame", self.current_time / self.delta)
            return
        
        # Otherwise we interpolate the border width
        initial_width = self._current_border_width
        start_frame = self.current_time / self.delta
        end_frame = (self.current_time + time) / self.delta
        self.log(f"Fading border width from {initial_width} to {width} from frame {start_frame} to {end_frame}")

        def f(frac):
            w = initial_width + frac * (width - initial_width)
            return supe.set_border_width(w)
        self.timeline.add_fraction_updater(f, start_frame, end_frame)
        self.current_time += time
        self._current_border_width = width

    def set_border_color(self, color, time=None):
        """
        Set the color of the border.
        If time is specified, the border will fade to the new color over that time.
        """
        supe = super()
        
        # If this is an instant change we add a transition function
        # at the current time
        if time is None:
            def f():
                return supe.set_border_color(color)
            self.timeline.add_transition(f, self.current_time / self.delta)
            self.log("Setting border color to", color, "at frame", self.current_time / self.delta)
            self._current_border_color = color
            return

        # Otherwise we create a colormap that fades from the initial color to the new color
        # and add a fraction updater to the timeline
        initial_color = self._current_border_color
        colormap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "border_fade", [initial_color, color], N=256
        )

        start_frame = self.current_time / self.delta
        end_frame = (self.current_time + time) / self.delta
        self.log(f"Fading border color from {initial_color} to {color} from frame {start_frame} to {end_frame}")

        def f(frac):
            intermediate_color = colormap(frac)
            return supe.set_border_color(intermediate_color)
        self.timeline.add_fraction_updater(f, start_frame, end_frame)
        self.current_time += time
        self._current_border_color = color


    def set_title(self, title, **kwargs):
        def f(t, **kwargs):
            return [self.ax.set_title(t, **kwargs)]

        self.timeline.add_transition(f, self.current_time / self.delta, title, **kwargs)

    def hide(self, object):
        if (not hasattr(object, "set_visible")) and (not hasattr(object, "show")):
            raise ValueError("Object must have set_visible or hide method")

        def f(obj):
            if hasattr(obj, "set_visible"):
                obj.set_visible(False)
                return [obj]
            else:
                return obj.hide()

        self.timeline.add_transition(f, self.current_time / self.delta, object)

    def show(self, object):
        if (not hasattr(object, "set_visible")) and (not hasattr(object, "show")):
            raise ValueError("Object must have set_visible or show method")

        def f(obj):
            if hasattr(obj, "set_visible"):
                obj.set_visible(True)
                return [obj]
            else:
                return obj.show()

        self.timeline.add_transition(f, self.current_time / self.delta, object)

    def hide_all_journeys(self):
        journeys = self.journeys[:]

        def f():
            out = []
            for journey in journeys:
                for j in journey:
                    if hasattr(j, "set_visible"):
                        j.set_visible(False)
                        out.append(j)
            return out

        self.timeline.add_transition(f, self.current_time / self.delta)

    def set_label_offsets(self, new_offsets, time):
        frame0 = self.current_time / self.delta
        frame1 = (self.current_time + time) / self.delta
        priself.lognt(
            f"Updating label locations from time {self.current_time} to {self.current_time + time} == frame {frame0} to {frame1}"
        )
        locations = {name: self.locations[name] for name in new_offsets}
        old_offsets = self._current_label_offsets.copy()

        def f(frac, new_offsets):
            out = []
            for name, offset in new_offsets.items():
                x0, y0 = locations[name]
                x0 = x0 + old_offsets[name][0]
                y0 = y0 + old_offsets[name][1]
                label = self.labels[name]
                y1, x1 = self.locations[name]
                x1 += offset[0]
                y1 += offset[1]
                x = x0 + frac * (x1 - x0)
                y = y0 + frac * (y1 - y0)
                label.set_position((x, y))
                out.append(label)
            return out

        self.timeline.add_fraction_updater(f, frame0, frame1, new_offsets)
        self.current_time += time
        self._current_label_offsets.update(new_offsets)

    def pulsing_circle(self, x, y, radius, radius_variation, time, npulse, **kwargs):
        frame0 = self.current_time / self.delta
        frame1 = (self.current_time + time) / self.delta
        self.log(
            f"Pulsing circle from time {self.current_time} to {self.current_time + time} == frame {frame0} to {frame1}"
        )
        c = plt.Circle((x, y), radius, **kwargs)
        self.ax.add_patch(c)
        c.set_visible(False)

        def f(frac):
            dr = radius_variation * np.sin(2 * np.pi * frac * npulse)
            c.set_visible(True)
            c.set_radius(radius * (1 + dr))
            return [c]

        self.timeline.add_fraction_updater(f, frame0, frame1)
        self.current_time += time
        return c
    
    def add_danger_pulse(self, lat, lon, radius, alpha_max, time):
        """
        Add a pulsing circle to indicate danger at a location.
        """
        # create a gaussian image
        y, x = np.mgrid[-4.0 : 4.0 : 0.01, -4.0 : 4.0 : 0.01]
        nx, ny = x.shape
        z = np.exp(-0.5*(x**2 + y**2))
        z = z / z.max()
        extent = [lon - 4*radius, lon + 4*radius, lat - 4*radius, lat + 4*radius]
        d = np.zeros((ny, nx, 4), dtype=np.float32)
        d[:, :, 0] = 1.0 # purely red
        d[:, :, 3] = 0.0 # initially fully transparent
        img = self.ax.imshow(d, extent=extent, transform=PC, zorder=zorders.PULSE)

        frame0 = self.current_time / self.delta
        frame1 = (self.current_time + time) / self.delta
        self.log(f"Will animate danger pulse from time {self.current_time} to {self.current_time + time} == frame {frame0} to {frame1}")
        def f(frac):
            # fade in then out again as a sine wave rising and falling
            alpha = np.sin(np.pi * frac)
            d[:, :, 3] = alpha * z * alpha_max
            img.set_data(d)
            img.set_extent(extent)
            img.set_transform(PC)
            return [img]
        self.timeline.add_fraction_updater(f, frame0, frame1)
        self.current_time += time


    def wait(self, time):
        self.current_time += time

    def rewind(self, time):
        self.current_time -= time

    def zoom(self, lat_min, lon_min, lat_max, lon_max, time):
        t0 = self.current_time
        current_lon_min, current_lon_max, current_lat_min, current_lat_max = self._current_extent
        start_x = (current_lon_min, current_lon_max)
        start_y = (current_lat_min, current_lat_max)
        frame0 = t0 / self.delta
        t1 = t0 + time
        frame1 = t1 / self.delta
        end_x = (lon_min, lon_max)
        end_y = (lat_min, lat_max)
        self.log(
            f"Zoom from x={start_x} and x={end_x} to x={end_x} and y={end_y} will take from time {t0} to {t1} == frame {frame0} to {frame1}"
        )
        self.timeline.add_fraction_updater(
            anim.zoom_to, frame0, frame1, self.ax, [start_x, start_y], [end_x, end_y]
        )

        self.timeline.add_every_frame_between_updater(
            self.tile_suite.redraw,
            frame0,
            frame1,
        )

        # if the fog is animated it will be updating anyway
        # and we don't need to add an update to it.
        if (self.fog is not None) and (not self.fog.animated):
            self.timeline.add_every_frame_between_updater(
                self.fog.redraw,
                frame0,
                frame1,

            )

        self.current_time = t1
        self._current_extent = (lon_min, lon_max, lat_min, lat_max)

    def fade_between_tiles(self, level0, level1, time):
        start_frame = self.current_time / self.delta
        end_frame = (self.current_time + time) / self.delta
        self.log(f"Fade between tile levels {level0}-{level1} from time {self.current_time} to {self.current_time + time}")

        def f(frac):
            self.tile_suite.set_alpha(level0, 1 - frac)
            self.tile_suite.set_alpha(level1, frac)
            return self.tile_suite.redraw()

        self.timeline.add_fraction_updater(f, start_frame, end_frame)
        self.current_time += time

    def fade_out_tiles(self, level, time):
        start_frame = self.current_time / self.delta
        end_frame = (self.current_time + time) / self.delta
        self.current_time += time
        self.log(f"Fade out tiles from time {self.current_time} to {self.current_time + time}")

        def f(frac):
            self.tile_suite.set_alpha(level, 1 - frac)
            return self.tile_suite.redraw()

        self.timeline.add_fraction_updater(f, start_frame, end_frame)

    def fade_in_tiles(self, level, time):
        start_frame = self.current_time / self.delta
        end_frame = (self.current_time + time) / self.delta
        self.log(f"Fade in tiles from time {self.current_time} to {self.current_time + time}")
        self.current_time += time

        def f(frac):
            self.tile_suite.set_alpha(level, frac)
            return self.tile_suite.redraw()

        self.timeline.add_fraction_updater(f, start_frame, end_frame)


    def unmask_point(self, lat, lon, r, time):
        frame0 = self.current_time / self.delta
        frame1 = (self.current_time + time) / self.delta
        self.log(
            f"Unmasking point from time {self.current_time} to {self.current_time + time} == frame {frame0} to {frame1}"
        )
        supe = super()

        def f(frac):
            return supe.unmask_point(lat, lon, r * frac)

        self.timeline.add_fraction_updater(f, frame0, frame1)
        self.current_time += time

    def unmask_location(self, place, r, time):
        frame0 = self.current_time / self.delta
        frame1 = (self.current_time + time) / self.delta
        self.log(
            f"Unmasking point from time {self.current_time} to {self.current_time + time} == frame {frame0} to {frame1}"
        )
        supe = super()

        def f(frac):
            return supe.unmask_location(place, r * frac)

        self.timeline.add_fraction_updater(f, frame0, frame1)
        self.current_time += time


    def hide_fog_of_war(self):
        supe = super()

        def f():
            return supe.hide_fog_of_war()

        self.timeline.add_transition(f, self.current_time / self.delta)

    def reset_fog_of_war(self):
        supe = super()

        def f():
            return supe.reset_fog_of_war()

        self.timeline.add_transition(f, self.current_time / self.delta)


    def add_ripple(self, lat, lon, sigma, scaling, time, iterations_per_frame=30, path_steps_per_frame=5, damping=0.99):
        frame_start = int(self.current_time / self.delta)
        frame_end = int(frame_start + time / self.delta)
        nx, ny = self.fig.get_size_inches() * self.fig.get_dpi()
        nx = int(nx)
        ny = int(ny)

        path = np.zeros((len(lat), 2))
        path[:, 0] = lon
        path[:, 1] = lat

        path = self.ax.transData.transform(path)
        path_xpix = path[:, 0]
        path_ypix = path[:, 1]


        wave_kwargs = dict(
                nx=nx, 
                ny=ny,
                Lx=nx/6,
                Ly=ny/6,
                g=10.0,
                h=100.0,
                dt=0.05,
                damping=damping,
        )
        self.wait(time)



        transform = DuckTransform(
            frame_start=frame_start,
            frame_end=frame_end,
            iterations_per_frame=iterations_per_frame,
            wave_kwargs=wave_kwargs,
            path_sigma=sigma,
            path_steps_per_frame=path_steps_per_frame,
            path_x=path_xpix,
            path_y=path_ypix,
            scaling=scaling,
        )
        self.fig.post_transform.add_transform(transform)
