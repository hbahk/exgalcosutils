#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 20 15:10:11 2022

@author: Hyeonguk Bahk
"""

import numpy as np
from matplotlib import pyplot as plt
from astropy.coordinates import SkyCoord
from astropy import unit as u

def match_catalogs(cat_lowres, cat_highres, tol):
    '''
    Match two catalogs from different astronomical surveys, based on their
    celestrial positions.
    
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
        The matching tolerence of angular separation. If this is 

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
    >>> psz_coord = SkyCoord(ra=psz['RAdeg'], dec=psz['DEdeg'], unit='deg')
    >>> xcls_coord = SkyCoord(ra=xcls['RAdeg'], dec=xcls'DEdeg'], unit='deg')
    
    Then we can carry out positional catalog matching:
        
    >>> ipsz, ixcls = match_catalogs(psz_coord, xcls_coord, tol=7*u.arcmin)
    >>> psz_matched, xcls_matched = psz[ipsz], xcls[ixcls]
    
    Note that we set psz_coord as a catalog of low resolution. The FWHM of
    XMM-Newton, which XCLASS catalog is based on, is ~6 arcsec and the FWHM of
    Planck data is ~ 7 arcmin. The `tol` parameter were set to be 7 
    arcmin, since it is apparently larger than 6 arcsec.
    
    '''
    
    if not (isinstance(cat_lowres, SkyCoord)
            and isinstance(cat_highres, SkyCoord)):
        raise ValueError('`cat_lowres` and `cat_highres` should be '+
                         'SkyCoord objects')
    
    if not isinstance(tol, u.quantity.Quantity):
        if type(tol) is float or type(tol) is int:
            tol = tol * u.arcsec
        else:
            raise ValueError('`tol` should be either a number (int or float)'+
                             ' or a Quantity object with given unit')
    
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
    # cmil ('c'ross 'm'atched 'i'ndex for 'l'ow resolution catalog)
    cmil, _, _ = cat_highres_matched.match_to_catalog_sky(cat_lowres)

    idxl = il[cmil]    # matched index of cat_lowres
    
    return idxl, idxh
    

def plot_offset_dist(c1, c2, tol):
    
    if not (isinstance(c1, SkyCoord) and isinstance(c2, SkyCoord)):
        raise ValueError('`c1` and `c2` should be SkyCoord objects')
    
    if not isinstance(tol, u.quantity.Quantity):
        if type(tol) is float or type(tol) is int:
            tol = tol * u.arcsec
        else:
            raise ValueError('`tol` should be either a number (int or float)'+
                             ' or a Quantity object with given unit')
    
    dra, ddec = c2.spherical_offsets_to(c1)
    
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    
    ax.plot(dra.arcmin, ddec.arcmin, '.', c='k')
    
    circ = plt.Circle((0,0), tol.value, fill=False, ec='k')
    ax.add_patch(circ)
    
    ax.axhline(0, c='k', ls=':', lw=1)
    ax.axvline(0, c='k', ls=':', lw=1)
    
    ax.text(0.98, 0.02, f'TOL={tol:.2f}',
            va='bottom', ha='right', transform=ax.transAxes)
    
    ax.set_xlabel(r'$\Delta \alpha$ ' + f'[{tol.unit.name}]')
    ax.set_ylabel(r'$\Delta \delta$ ' + f'[{tol.unit.name}]')
    ax.axis('equal')
    
    ax.minorticks_on()
    ax.tick_params(direction='in', top=True, right=True)
    ax.tick_params(which='minor', direction='in', top=True, right=True)
    
    plt.tight_layout()
    plt.show()
