#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:10:11 2022

@author: Hyeonguk Bahk
"""

import numpy as np
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import units as u

__all__ = ["match_catalogs", "plot_offset_dist"]

def match_catalogs(cat_lowres, cat_highres, tol):
    """
    Cross-match two catalogs from different astronomical surveys, based on
    their celestrial positions.
    
    Returns the ordered index array of matched sources for the given catalogs.
    The output index array are sorted by original cat_highres index.

    Parameters
    ----------
    cat_lowres : SkyCoord array
        A position array of a catalog to match with the other. This should be
        based on survey with lower spatial resolution (larger FWHM) than the
        other.
    cat_highres : SkyCoord array
        A position array of a catalog to match with the cat_lowres, the first
        parameter. This should be based on survey with higher spatial
        resolution (smaller FWHM) than the other.
    tol : float, int or Quantity with angular unit.
        The matching tolerance of angular separation. If this is provided
        without angluar unit, then this quantity will be condidered as arcsec
        by default.

    Returns
    -------
    idxl : TYPE
        An Index array of matched sources for the cat_lowres (first parameter)
    idxh : TYPE
        An Index array of matched sources for the cat_highres (second
        parameter).
        
    Examples
    --------
    Let's consider we have Planck Sunyaev-Zeldovich catalog for SZ-selected 
    galaxy clusters and XCLASS catalog for X-ray-selected galaxy clusters.
    After loading these catalogs in tables (such as astropy table, which is
    the case in this example) with names of psz and xcls. Then we need to make
    SkyCoord array with RA and DEC information listed on the tables:
        
    >>> from astropy.coordinates import SkyCoord
    >>> from astropy import units as u
    >>> psz_coord = SkyCoord(ra=psz["RAdeg"], dec=psz["DEdeg"], unit="deg")
    >>> xcls_coord = SkyCoord(ra=xcls["RAdeg"], dec=xcls"DEdeg"], unit="deg")
    
    Then we can carry out positional catalog matching:
        
    >>> ipsz, ixcls = match_catalogs(psz_coord, xcls_coord, tol=7*u.arcmin)
    >>> psz_matched, xcls_matched = psz[ipsz], xcls[ixcls]
    
    Note that we set psz_coord as a catalog of low resolution. The FWHM of
    XMM-Newton, which XCLASS catalog is based on, is ~6 arcsec and the FWHM of
    Planck data is ~ 7 arcmin. The `tol` parameter were set to be 7 
    arcmin, since it is apparently larger than 6 arcsec.
    
    """
    
    if not (isinstance(cat_lowres, SkyCoord)
            and isinstance(cat_highres, SkyCoord)):
        raise ValueError("`cat_lowres` and `cat_highres` should be "+
                         "SkyCoord objects")
    
    if not isinstance(tol, u.quantity.Quantity):
        if type(tol) is float or type(tol) is int:
            tol = tol * u.arcsec
        else:
            raise ValueError("`tol` should be either a number (int or float)"+
                             " or a Quantity object with given unit")
    
    # save original order
    ih = np.arange(len(cat_highres))    # original index of cat_highres
    il = np.arange(len(cat_lowres))     # original index of cat_lowhres
    
    # 1st match within `tol` in single direction
    idx, sep, _ = cat_lowres.match_to_catalog_sky(cat_highres)
    
    sep_condi = sep < tol   # separation condition
    
    idx_unique = np.unique(idx[sep_condi])  # rules out duplications in highres
    cat_highres_matched = cat_highres[idx_unique] # not ordered by lowres
    idxh = ih[idx_unique]   # matched index of cat_highres
    
    # 2nd match with pre-matched catalog where all duplications were ruled out
    # cmil ("c"ross "m"atched "i"ndex for "l"ow resolution catalog)
    cmil, _, _ = cat_highres_matched.match_to_catalog_sky(cat_lowres)

    idxl = il[cmil]    # matched index of cat_lowres
    
    return idxl, idxh
#%%    

def plot_offset_dist(c1, c2, tol, ax=None, hist=False,
                     plot_kwargs={}, hist_kwargs={},
                     circ_kwargs={}, text_kwargs={}, return_axes=False):
    """
    Plot a offset distribution figure for positions of two matched catalogs.

    Parameters
    ----------
    c1 : SkyCoord array
        SkyCoord coordiantes of a matched catalog.
    c2 : SkyCoord array
        SkyCoord coordiantes of the other matched catalog.
    tol : float, int or Quantity with angular unit.
        The matching tolerance of angular separation. If this is provided
        without angluar unit, then this quantity will be condidered as arcsec
        by default.
    hist : bool, optional
        If True then draw histograms for each axis (longitudinal and
        latitudinal) on the corresponding side axis. The default is False.
    plot_kwargs : dict, optional
        kwargs for main plot (ax.plot(**kwargs)). The default is {}.
    hist_kwargs : dict, optional
        kwargs for hist plot (ax.hist(**kwargs)). The default is {}.
    circ_kwargs : dict, optional
        kwargs for circle patch (plt.Circle(**kwargs)). The default is {}.
    text_kwargs : dict, optional
        kwargs for text in main plot (ax.text(**kwargs)). The default is {}.
    return_axes : TYPE, optional
        If True all AxesSubplot will be returned. In case of hist=False, single
        main subplot will be returned, otherwise the axes will be returned in
        form of following:
            array([[<AxesSubplot:>, <AxesSubplot:>],
                   [<AxesSubplot:>, <AxesSubplot:>]], dtype=object)
        The default is False.

    Returns
    -------
    (Optional) AxesSubplot or array of AxesSubplot, if return_axis=True

    """
    if not (isinstance(c1, SkyCoord) and isinstance(c2, SkyCoord)):
        raise ValueError("`c1` and `c2` should be SkyCoord objects")
    
    if not isinstance(tol, u.quantity.Quantity):
        if type(tol) is float or type(tol) is int:
            tol = tol * u.arcsec
        else:
            raise ValueError("`tol` should be either a number (int or float)"+
                             " or a Quantity object with given unit")
    
    dra, ddec = c2.spherical_offsets_to(c1)
    dra, ddec = dra.to(tol.unit), ddec.to(tol.unit)
    
    if hist:
        if ax:
            print('WARNING: `ax` and `hist` are not compatible.'+
                  ' `ax` option will be ignored.')
        fig, axes = plt.subplots(2, 2, figsize=(7,7),
                                 gridspec_kw={"width_ratios": [4, 1],
                                              "height_ratios": [1, 4]})
        (hhax, nax), (ax, vhax) = axes
    elif ax == None:
        fig, ax = plt.subplots(1, 1, figsize=(7,7))
    
    # main scatter plot
    # - default kwargs setting (fmt=".", c="k")
    marker = plot_kwargs.pop("marker", ".")
    ls_keyname = "ls" if "ls" in plot_kwargs.keys() else "linestyle"
    plot_ls = plot_kwargs.pop(ls_keyname, "")
    color_keyname = "c" if "c" in plot_kwargs.keys() else "color" # for alias
    plot_c = plot_kwargs.pop(color_keyname, "k")
    
    # - plotting positional offset
    ax.plot(dra.value, ddec.value, marker=marker, ls=plot_ls, c=plot_c,
            **plot_kwargs)
    
    # - default kwargs for the tolerance circle (fill=False, ec="k", zorder=0)
    circ_fill = circ_kwargs.pop("fill", False)
    ec_keyname = "ec" if "ec" in circ_kwargs.keys() else "edgecolor"
    circ_ec = circ_kwargs.pop(ec_keyname, "k")
    circ_zorder = circ_kwargs.pop("zorder", 0)
    
    # - drawing the circle
    circ = plt.Circle((0,0), tol.value, fill=circ_fill, ec=circ_ec,
                      zorder=circ_zorder, **circ_kwargs)
    ax.add_patch(circ)
    
    # - making x and y scale equal
    # ax.set_aspect("equal") # This doesn"t work well
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    lim = np.array([-1,1]) * np.max(np.abs(np.concatenate((xlim, ylim))))
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    
    # - horizontal and vertical line passing the zeropoint
    ax.axhline(0, c="k", ls=":", lw=1)
    ax.axvline(0, c="k", ls=":", lw=1)
    
    # - default kwargs for the text
    # +- (x=0.99, y=0.01, va="bottom", ha="right", transform=ax.transAxes)
    text_x, text_y = text_kwargs.pop("x", 0.99), text_kwargs.pop("y", 0.01)
    va_keyname = "va" if "va" in text_kwargs.keys() else "verticalalignment"
    ha_keyname = "ha" if "ha" in text_kwargs.keys() else "horizontalignment"
    text_va = text_kwargs.pop(va_keyname,"bottom")
    text_ha = text_kwargs.pop(ha_keyname, "right")
    text_transform = text_kwargs.pop("transform", ax.transAxes)
    
    # - add information about the tolerance
    ax.text(text_x, text_y, f"TOL={tol:.1f}",
            va=text_va, ha=text_ha, transform=text_transform, **text_kwargs)
    
    # - axis label
    ax.set_xlabel(r"$\Delta \alpha$ " + f"[{tol.unit.name}]")
    ax.set_ylabel(r"$\Delta \delta$ " + f"[{tol.unit.name}]")
    
    # - changing tick direction and adding minor ticks
    ax.minorticks_on()
    ax.tick_params(direction="in", top=True, right=True)
    ax.tick_params(which="minor", direction="in", top=True, right=True)
    
    # histograms on side axes
    if hist:
        # - unuse top-right axis
        nax.axis("off")
        
        # - default kwargs for histogram (bins=20, histtype="step", color="k")
        bins = hist_kwargs.pop("bins", 20)
        histtype = hist_kwargs.pop("histtype", "step")
        hist_color = hist_kwargs.pop("color", "k")
        
        # - drawing histogram
        _ = hhax.hist(dra.value, bins=bins, histtype=histtype,
                      color=hist_color, **hist_kwargs)
        _ = vhax.hist(ddec.value, bins=bins, histtype=histtype,
                      color=hist_color, orientation="horizontal",
                      **hist_kwargs)
        
        hhax.get_shared_x_axes().join(ax, hhax)
        vhax.get_shared_y_axes().join(ax, vhax)
        
        hhax.set_xlim(lim)
        vhax.set_ylim(lim)
        
        for axis, lb, ll in zip([hhax, vhax], [False, True], [True, False]):

            axis.minorticks_on()
            axis.tick_params(direction="in", top=True, right=True,
                             labelbottom=lb, labelleft=ll)
            axis.tick_params(which="minor", direction="in",
                             top=True, right=True)
            
            plt.draw()    # to populate tick labels
            lab_f = axis.get_yticklabels if ll==True else axis.get_xticklabels
            plt.setp(lab_f()[0], visible=False)
        
    plt.tight_layout()
    plt.subplots_adjust(hspace=0, wspace=0)
    plt.show()
    
    if return_axes:
        return axes if hist else ax
