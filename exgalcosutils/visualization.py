from types import MethodType
import warnings

import astropy.units as u
import numpy as np
from astropy.coordinates import ICRS, SkyCoord, Angle
from matplotlib import pyplot as plt
from mocpy import MOC, WCS
from astropy.visualization.wcsaxes.frame import EllipticalFrame
from matplotlib import patheffects


def init_sky(
    projection="mollweide",
    ra_center=120,
    galactic_plane_color="k",
    ecliptic_plane_color="red",
    ax=None,
):
    """Initialize matplotlib axes with a projection of the full sky.
    Sourced from desiutil_plots.
    Parameters
    ----------
    projection : :class:`str`, optional
        Projection to use. Defaults to 'mollweide'.  To show the available projections,
        call :func:`matplotlib.projections.get_projection_names`.
    ra_center : :class:`float`, optional
        Projection is centered at this RA in degrees. Default is +120°.
    galactic_plane_color : color name, optional
        Draw a solid curve representing the galactic plane using the specified color, or
        do nothing when ``None``.
    ecliptic_plane_color : color name, optional
        Draw a dotted curve representing the ecliptic plane using the specified color, or
        do nothing when ``None``.
    ax : :class:`~matplotlib.axes.Axes`, optional
        Axes to use for drawing this map, or create new axes if ``None``.
    Returns
    -------
    :class:`~matplotlib.axes.Axes`
        A matplotlib Axes object.  Helper methods ``projection_ra()`` and
        ``projection_dec()`` are added to the object to facilitate conversion to
        projection coordinates.
    """

    # Internal functions.
    def projection_ra(ra):
        r"""Shift `ra` to the origin of the Axes object and convert to radians.
        Parameters
        ----------
        ra : array-like
            Right Ascension in degrees.
        Returns
        -------
        array-like
            `ra` converted to plot coordinates.
        Notes
        -----
        In matplotlib, map projections expect longitude (RA), latitude (Dec)
        in radians with limits :math:`[-\pi, \pi]`, :math:`[-\pi/2, \pi/2]`,
        respectively.
        """
        # Shift RA values.
        r = np.remainder(ra + 360 - ra_center, 360)
        # Scale conversion to [-180, 180].
        r[r > 180] -= 360
        # Reverse the scale: East to the left.
        r = -r
        return np.radians(r)

    def projection_dec(dec):
        """Shift `dec` to the origin of the Axes object and convert to radians.
        Parameters
        ----------
        dec : array-like
            Declination in degrees.
        Returns
        -------
        array-like
            `dec` converted to plot coordinates.
        """
        return np.radians(dec)

    # Create ax.
    if ax is None:
        fig = plt.figure(figsize=(10.0, 5.0), dpi=100)
        ax = fig.add_subplot(111, projection=projection)

    # Prepare labels.
    base_tick_labels = np.array([150, 120, 90, 60, 30, 0, 330, 300, 270, 240, 210])
    base_tick_labels = np.remainder(base_tick_labels + 360 + ra_center, 360)
    tick_labels = np.array(["{0}°".format(lab) for lab in base_tick_labels])

    # Galactic plane.
    if galactic_plane_color is not None:
        galactic_l = np.linspace(0, 2 * np.pi, 1000)
        galactic = SkyCoord(
            l=galactic_l * u.radian,
            b=np.zeros_like(galactic_l) * u.radian,
            frame="galactic",
        ).transform_to(ICRS)
        galup = SkyCoord(
            l=galactic_l * u.radian,
            b=np.ones_like(galactic_l) * 10 * u.deg,
            frame="galactic",
        ).transform_to(ICRS)
        galdo = SkyCoord(
            l=galactic_l * u.radian,
            b=np.ones_like(galactic_l) * (-10) * u.deg,
            frame="galactic",
        ).transform_to(ICRS)

        idx_sort = np.argsort(
            galactic.ra.wrap_at(ra_center * u.deg - 180 * u.deg).degree
        )
        galactic = galactic[idx_sort]
        idx_sort_up = np.argsort(
            galup.ra.wrap_at(ra_center * u.deg - 180 * u.deg).degree
        )
        galup = galup[idx_sort_up]
        idx_sort_do = np.argsort(
            galdo.ra.wrap_at(ra_center * u.deg - 180 * u.deg).degree
        )
        galdo = galdo[idx_sort_do]

        # Project to map coordinates and display.
        ax.plot(
            ax.projection_ra(galactic.ra.degree),
            ax.projection_dec(galactic.dec.degree),
            lw=1,
            alpha=0.75,
            c=galactic_plane_color,
            zorder=20,
        )
        ax.plot(
            ax.projection_ra(galup.ra.degree),
            ax.projection_dec(galup.dec.degree),
            lw=1,
            ls="--",
            alpha=0.75,
            c=galactic_plane_color,
            zorder=20,
        )
        ax.plot(
            ax.projection_ra(galdo.ra.degree),
            ax.projection_dec(galdo.dec.degree),
            lw=1,
            ls="--",
            alpha=0.75,
            c=galactic_plane_color,
            zorder=20,
        )

    # Ecliptic plane.
    if ecliptic_plane_color is not None:
        ecliptic_l = np.linspace(0, 2 * np.pi, 50)
        ecliptic = SkyCoord(
            lon=ecliptic_l * u.radian,
            lat=np.zeros_like(ecliptic_l) * u.radian,
            distance=1 * u.Mpc,
            frame="heliocentrictrueecliptic",
        ).transform_to(ICRS)
        idx_sort = np.argsort(
            ecliptic.ra.wrap_at(ra_center * u.deg - 180 * u.deg).degree
        )
        ecliptic = ecliptic[idx_sort]

        # Project to map coordinates and display.
        ax.plot(
            ax.projection_ra(ecliptic.ra.degree),
            ax.projection_dec(ecliptic.dec.degree),
            lw=1,
            ls=":",
            alpha=0.75,
            c=ecliptic_plane_color,
            zorder=20,
        )

    # Set RA labels.
    labels = ax.get_xticklabels()
    for lab, item in enumerate(labels):
        item.set_text(tick_labels[lab])
    ax.set_xticklabels(labels)

    # Set axis labels.
    ax.set_xlabel("RA [deg]")
    ax.set_ylabel("Dec [deg]")
    ax.grid(True)

    # Attach helper methods.
    if hasattr(ax, "_ra_center"):
        warnings.warn("Attribute '_ra_center' detected.  Will be overwritten!")
    ax._ra_center = ra_center
    if hasattr(ax, "projection_ra"):
        warnings.warn("Attribute 'projection_ra' detected.  Will be overwritten!")
    ax.projection_ra = MethodType(projection_ra, ax)
    if hasattr(ax, "projection_dec"):
        warnings.warn("Attribute 'projection_dec' detected.  Will be overwritten!")
    ax.projection_dec = MethodType(projection_dec, ax)
    return ax


def init_sky_moc(
    projection="MOL",
    ra_center=120,
    galactic_plane_color="red",
    ecliptic_plane_color="red",
):
    """Initialize matplotlib axes with a projection of the full sky, using WCS
    projection from mocpy.
    
    Parameters
    ----------
    projection : :class:`str`, optional
        Projection to use. Defaults to 'MOL'.  To show the available projections,
        see https://docs.astropy.org/en/stable/wcs/supported_projections.html.
    ra_center : :class:`float`, optional
        Projection is centered at this RA in degrees. Default is +120°.
    galactic_plane_color : color name, optional
        Draw a solid curve representing the galactic plane using the specified color, or
        do nothing when ``None``.
    ecliptic_plane_color : color name, optional
        Draw a dotted curve representing the ecliptic plane using the specified color, or
        do nothing when ``None``.
        
    Returns
    -------
    :class:`~matplotlib.axes.Axes`
        A matplotlib Axes object.
    """
    fig = plt.figure(figsize=(10.0, 5.0), dpi=100)
    wcs = WCS(fig=fig,
            fov=160*u.deg,
            center=SkyCoord(ra_center, 0, unit='deg'),
            coordsys="icrs",
            rotation=Angle(0, u.deg),
            projection=projection #AIT
            ).w
    ax = fig.add_subplot(111, projection=wcs, frame_class=EllipticalFrame)
    ax = init_sky(ax=ax, ra_center=ra_center, galactic_plane_color=None, ecliptic_plane_color=None)
    
    base_tick_labels = np.array([120, 60, 0, 300, 240])
    base_tick_labels = np.remainder(base_tick_labels+360+ra_center, 360)
    ax.coords[0].set_ticks(base_tick_labels * u.deg)
    ax.coords[0].set_major_formatter('dd')
    path_effects=[patheffects.withStroke(linewidth=3, foreground='w', alpha=0.7)]
    ax.coords[0].set_ticklabel(color='k', path_effects=path_effects)

    base_yticks = np.array([-60, -30, 0, 30, 60])
    base_yticks = np.array([-45, 0, 45])
    ax.coords[1].set_ticks(base_yticks * u.deg)
    ax.coords[1].set_major_formatter('dd')

    ax.grid(True)
    
    # Draw the galactic plane
    if galactic_plane_color is not None:
        galactic_l = np.linspace(0, 2 * np.pi, 1000)
        galactic = SkyCoord(l=galactic_l*u.radian, b=np.zeros_like(galactic_l)*u.radian,
                            frame='galactic').transform_to(ICRS)
        galup = SkyCoord(l=galactic_l*u.radian, b=np.ones_like(galactic_l)*10*u.deg,
                            frame='galactic').transform_to(ICRS)
        galdo = SkyCoord(l=galactic_l*u.radian, b=np.ones_like(galactic_l)*(-10)*u.deg,
                            frame='galactic').transform_to(ICRS)
        
        idx_sort = np.argsort(galactic.ra.wrap_at(ra_center*u.deg-180*u.deg).degree)
        galactic = galactic[idx_sort]
        idx_sort_up = np.argsort(galup.ra.wrap_at(ra_center*u.deg-180*u.deg).degree)
        galup = galup[idx_sort_up]
        idx_sort_do = np.argsort(galdo.ra.wrap_at(ra_center*u.deg-180*u.deg).degree)
        galdo = galdo[idx_sort_do]
        
        # Project to map coordinates and display.
        ax.plot(galactic.ra.degree,
                        galactic.dec.degree,
                        lw=1, alpha=0.75, c=galactic_plane_color, zorder=20,
                        transform=ax.get_transform('world'))
        ax.plot(galup.ra.degree,
                        galup.dec.degree,
                        lw=1, ls='--', alpha=0.75, c=galactic_plane_color, zorder=20,
                        transform=ax.get_transform('world'))
        ax.plot(galdo.ra.degree,
                        galdo.dec.degree,
                        lw=1, ls='--', alpha=0.75, c=galactic_plane_color, zorder=20,
                        transform=ax.get_transform('world'))
        
    # Draw the ecliptic plane
    if ecliptic_plane_color is not None:
        ecliptic_l = np.linspace(0, 2 * np.pi, 50)
        ecliptic = SkyCoord(lon=ecliptic_l*u.radian, lat=np.zeros_like(ecliptic_l)*u.radian,
                            distance=1*u.Mpc, frame='heliocentrictrueecliptic').transform_to(ICRS)
        idx_sort = np.argsort(ecliptic.ra.wrap_at(ra_center*u.deg-180*u.deg).degree)
        ecliptic = ecliptic[idx_sort]
        
        # Project to map coordinates and display.
        ax.plot(ecliptic.ra.degree,
                        ecliptic.dec.degree,
                        lw=1, ls=':', alpha=0.75, c=ecliptic_plane_color, zorder=20,
                        transform=ax.get_transform('world'))
        
    return ax
        