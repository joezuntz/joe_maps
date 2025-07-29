from .maps import Map
from . import anim
import matplotlib.patches
from .styles import mmc_colors, PC
import matplotlib.pyplot as plt
import numpy as np
from . import zorders

class AnimatedMap(Map):
    def __init__(self, *args, delta, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta = delta
        self.timeline = anim.Timeline(self.ax)
        self.init_rect = matplotlib.patches.Rectangle((2, 28), 0.01, 0.01, alpha=0, transform=PC)
        self.ax.add_patch(self.init_rect)
        self.current_time = 0.0
        # self._current_zoom = self.ax.get_xlim(), self.ax.get_ylim()
        self._current_extent = self.ax.get_extent(crs=PC)
        self._current_label_offsets = {}
        self.fog_img = None

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
            zorder=zorder,
            transform=PC,
        )
        self.labels[place] = label
        self._current_label_offsets[place] = offset
        label.set_visible(False)
        print(f"Showing label {place} at", self.current_time)
        self.show(label)
        return label
    
    def add_text(self, place, text, time=None, offset=(0, 0), facecolor=mmc_colors.wheat, textcolor=mmc_colors.dark_blue, edgecolor=mmc_colors.dark_blue, **kwargs):
        text = super().add_text(place, text, offset=offset, facecolor=facecolor, textcolor=textcolor, edgecolor=edgecolor, **kwargs)
        text.set_visible(False)
        print(f"Showing text '{text}' at {place} at", self.current_time)
        self.show(text)
        if time is not None:
            frame = (self.current_time + time) / self.delta
            print(f"Hiding text '{text}' at {place} at", self.current_time + time)
            self.timeline.add_transition(anim.make_disappear, frame, text)
            self.current_time += time
        return text
    

    def add_point(self, place, *args, **kwargs):
        points = super().add_point(place, *args, **kwargs)
        points.set_visible(False)
        print(f"Showing point at {place} at", self.current_time)
        self.show(points)
        return points

    def _animate_journey(self, journey, time, speed, fade_after=0):
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


        if self.fog is not None and journey.fog_clearance is not None:
            print("Setting up fog unmasking for journey")
            def unmask_journey(f):
                print("Unmasking journey at fraction", f)
                self.fog.unmask_journey(journey, frac=f)
                return self.fog.redraw()
            self.timeline.add_fraction_updater(unmask_journey, start_frame, end_frame)

        self.timeline.add_fraction_updater(journey.animate, start_frame, end_frame)

        # Add the animation to the timeline
        # self.timeline.add_fraction_updater(anim.wipe_journey, start_frame, end_frame, self.ax, direction, journey)
        self.current_time = end_time

    
        if fade_after > 0:
            fade_frame = (self.current_time + fade_after) / self.delta
            print(f"Journey will fade out from end to  {fade_frame:.1f}")
            self.timeline.add_fraction_updater(
                journey.fade_out, end_frame, fade_frame,
            )

    def add_journey(
        self, *args, time=None, speed=None, fade_after=0, **kwargs
    ):
        journey = super().add_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, fade_after=fade_after)
        return journey

    def add_return_journey(
        self, *args, time=None, speed=None, fade_after=0, **kwargs
    ):
        journey = super().add_return_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, fade_after=fade_after)
        return journey

    def add_gpx_journey(
        self, *args, speed=None, time=None, fade_after=0, **kwargs
    ):
        journey = super().add_gpx_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, fade_after=fade_after)
        return journey
    
    def add_band_journey(
        self, *args, time=None, speed=None,fade_after=0, **kwargs):
        journey = super().add_band_journey(*args, **kwargs)
        self._animate_journey(journey, time, speed, fade_after=fade_after)
        return journey

    def save(self, path, **kwargs):
        interval = self.delta * 1000
        frames = int(np.ceil(self.current_time / self.delta)) + 1
        print(f"Animation will have {frames} frames")
        self.timeline.save(self.fig, path, interval=interval, frames=frames, **kwargs)

    def delay_last_transition(self):
        self.timeline.transitions[-1][1] = self.current_time / self.delta

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
        print(
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
        print(f"Fade between tile levels {level0}-{level1} from time {self.current_time} to {self.current_time + time}")

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
        print(f"Fade out tiles from time {self.current_time} to {self.current_time + time}")

        def f(frac):
            self.tile_suite.set_alpha(level, 1 - frac)
            return self.tile_suite.redraw()

        self.timeline.add_fraction_updater(f, start_frame, end_frame)

    def fade_in_tiles(self, level, time):
        start_frame = self.current_time / self.delta
        end_frame = (self.current_time + time) / self.delta
        print(f"Fade in tiles from time {self.current_time} to {self.current_time + time}")
        self.current_time += time

        def f(frac):
            self.tile_suite.set_alpha(level, frac)
            return self.tile_suite.redraw()

        self.timeline.add_fraction_updater(f, start_frame, end_frame)


    # def add_animated_fog_of_war(self, nx, alpha=0.5):
    #     self.add_fog_of_war(nx, alpha=alpha, should_change_each_image=True)

    #     def update(i):
    #         return self.rebuild_fog_of_war()

    #     self.timeline.add_every_frame_updater(update)

    # def unmask_circle(self, x, y, r, time, **kwargs):
    #     frame0 = self.current_time / self.delta
    #     frame1 = (self.current_time + time) / self.delta
    #     print(
    #         f"Unmasking circle from time {self.current_time} to {self.current_time + time} == frame {frame0} to {frame1}"
    #     )
    #     supe = super()

    #     def f(frac):
    #         supe.unmask_circle(x, y, r * frac)
    #         return self.rebuild_fog_of_war()

    #     self.timeline.add_fraction_updater(f, frame0, frame1)
    #     self.current_time += time

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

    def add_fog(self, nx, alpha=0.5, animated=True, smoothing=0):
        fog = super().add_fog(nx, alpha=alpha, animated=animated, smoothing=smoothing)

        # if the fog is animated then we need to update it every frame
        # any of our journeys will also update the fog
        if fog.animated:
            # The frame argument is ignored by the fog
            # for now but I might want it later if we want
            # fancier fog effects
            def f(frame):
                # The reanimate argument is used because
                # we only want to update the fog once per
                # frame, not every time the fog is redrawn
                # due to a journey being unmasked
                return fog.redraw(reanimate=True)
            self.timeline.add_every_frame_updater(f)
