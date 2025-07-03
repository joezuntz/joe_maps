from .maps import Map
from . import anim
import matplotlib.patches
from .styles import mmc_colors
import matplotlib.pyplot as plt
import numpy as np


class AnimatedMap(Map):
    def __init__(self, *args, duration, delta, **kwargs):
        super().__init__(*args, **kwargs)
        frames = int(np.ceil(duration / delta))
        print(f"Animation will have {frames} frames")
        self.delta = delta
        self.duration = duration
        self.timeline = anim.Timeline(self.ax, frames=frames)
        self.init_rect = matplotlib.patches.Rectangle((2, 28), 0.01, 0.01, alpha=0)
        self.ax.add_patch(self.init_rect)
        self.current_time = 0.0
        self._current_zoom = self.ax.get_xlim(), self.ax.get_ylim()
        self._current_label_offsets = {}
        self.fog_img = None

    def add_label(
        self,
        place,
        offset=(0, 0),
        facecolor=mmc_colors.wheat,
        textcolor=mmc_colors.dark_blue,
        edgecolor=mmc_colors.dark_blue,
        **kwargs,
    ):
        y, x = self.locations[place]
        box = dict(boxstyle="round", facecolor=facecolor, alpha=1, edgecolor=edgecolor)
        label = self.ax.text(
            x + offset[0],
            y + offset[1],
            place,
            fontsize=14,
            bbox=box,
            fontname="Arial",
            color=textcolor,
        )
        self.labels[place] = label
        self._current_label_offsets[place] = offset
        label.set_visible(False)
        print(f"Showing label {place} at", self.current_time)
        self.show(label)
        return label

    def add_point(self, place, *args, **kwargs):
        points = super().add_point(place, *args, **kwargs)
        points.set_visible(False)
        print(f"Showing point at {place} at", self.current_time)
        self.show(points)
        return points

    def _animate_journey(self, journey, time, speed, unmask_radius=0):
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
        print(
            f"Journey will take from time {self.current_time:.2f} to {end_time:.2f} == frame {start_frame:.1f} to {end_frame:.1f}"
        )

        # # If there is a label, make it appear and disappear.  Really need a journey class to do this properly.
        # if len(journey) > 1 and journey[1] is not None:
        #     journey[1].set_visible(False)
        #     self.timeline.add_transition(anim.make_appear, start_frame, journey[1])
        #     self.timeline.add_transition(anim.make_disappear, end_frame, journey[1])

        # if (self.fog_img is not None) and unmask_radius:

        #     def unmask_journey(f):
        #         self.unmask_journey(journey, unmask_radius, frac=f)
        #         return self.rebuild_fog_of_war()
        #     self.timeline.add_fraction_updater(unmask_journey, start_frame, end_frame)

        self.timeline.add_fraction_updater(journey.animate, start_frame, end_frame)

        # Add the animation to the timeline
        # self.timeline.add_fraction_updater(anim.wipe_journey, start_frame, end_frame, self.ax, direction, journey)
        self.current_time = end_time

    def add_journey(
        self, *args, time=None, speed=None, direction=None, fog_clear_radius=0, **kwargs
    ):
        journey = super().add_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, fog_clear_radius)
        return journey

    def add_return_journey(
        self, *args, time=None, speed=None, direction=None, fog_clear_radius=0, **kwargs
    ):
        journey = super().add_return_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, fog_clear_radius)
        return journey

    def add_gpx_journey(
        self, *args, speed=None, time=None, direction=None, fog_clear_radius=0, **kwargs
    ):
        journey = super().add_gpx_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, fog_clear_radius)
        return journey

    def save(self, path, **kwargs):
        interval = self.delta * 1000
        self.timeline.save(self.fig, path, interval=interval, **kwargs)

    def delay_last_transition(self):
        self.timeline.transitions[-1][1] = self.current_time / self.delta

    def set_title(self, title, **kwargs):
        def f(t, **kwargs):
            return [self.ax.set_title(t, **kwargs)]

        self.timeline.add_transition(f, self.current_time / self.delta, title, **kwargs)

    def hide(self, object):
        def f(obj):
            obj.set_visible(False)
            return [obj]

        self.timeline.add_transition(f, self.current_time / self.delta, object)

    def show(self, object):
        def f(obj):
            obj.set_visible(True)
            return [obj]

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
        print(
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
        print(
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

    def wait(self, time):
        self.current_time += time

    def zoom(self, lat_min, lon_min, lat_max, lon_max, time):
        t0 = self.current_time
        start_x, start_y = self._current_zoom
        frame0 = t0 / self.delta
        t1 = t0 + time
        frame1 = t1 / self.delta
        end_x = (lon_min, lon_max)
        end_y = (lat_min, lat_max)
        print(
            f"Zoom to x={end_x} and y={end_y} will take from time {t0} to {t1} == frame {frame0} to {frame1}"
        )
        self.timeline.add_fraction_updater(
            anim.zoom_to, frame0, frame1, self.ax, [start_x, start_y], [end_x, end_y]
        )
        if self.fog_mask is not None:
            # zoom the fog of war too?
            self.timeline.add_fraction_updater(
                anim.zoom_extent,
                frame0,
                frame1,
                self.fog_img,
                [start_x, start_y],
                [end_x, end_y],
            )

        self.current_time = t1
        self._current_zoom = end_x, end_y

    def add_tiles(self, zoom_level, extent=None, tiler=None):
        if extent is None:
            xlim, ylim = self._current_zoom
            print("Will add tiles at", xlim, ylim, "at", self.current_time)
            extent = [xlim[0], xlim[1], ylim[0], ylim[1]]

        frame = self.current_time / self.delta
        supe = super()

        def f():
            return [supe.add_tiles(zoom_level, extent, tiler)]

        # only actually add the tiles later when we are at the
        # correct zoom
        self.timeline.add_transition(f, frame)

    def set_uncertain_title(
        self, titles, start_time, end_time, flicker_period, **kwargs
    ):
        """
        Flicker between possible titles
        """

        def f(t, **kwargs):
            pass

    def add_animated_fog_of_war(self, nx, alpha=0.5):
        self.add_fog_of_war(nx, alpha=alpha, should_change_each_image=True)

        def update(i):
            return self.rebuild_fog_of_war()

        self.timeline.add_every_frame_updater(update)

    def unmask_circle(self, x, y, r, time, **kwargs):
        frame0 = self.current_time / self.delta
        frame1 = (self.current_time + time) / self.delta
        print(
            f"Unmasking circle from time {self.current_time} to {self.current_time + time} == frame {frame0} to {frame1}"
        )
        supe = super()

        def f(frac):
            supe.unmask_circle(x, y, r * frac)
            return self.rebuild_fog_of_war()

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
