#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 16:24:00 2021

@author: hbahk

The Legacy Survey catalog retrieval from Sky Viewer of Legacy Survey
"""

import numpy as np
from astropy.io import fits
from astropy.table import Table, hstack
from astropy.coordinates import SkyCoord
from astropy.visualization import make_lupton_rgb
from astropy.wcs import WCS
from astropy.visualization.wcsaxes import SphericalCircle
from matplotlib import pyplot as plt
# from PIL import Image
from urllib.error import HTTPError
# from io import BytesIO
from exgalcosutils.catalog import match_catalogs
from astropy.utils.data import conf
import astropy.units as u

__all__ = ['get_lgs_image', 'get_lgs_image_lupton', 'get_lgs_galaxies',
           'get_info_fig']

def get_lgs_image(cra, cdec, size=1600, pix_scale=0.262, dr=10):
    """
    Retrieve a grz color cutout image near given RA/Dec from DESI Legacy Survey
    DR10 (or DR9).

    Parameters
    ----------
    cra, cdec : float or int in degree unit
        The central RA, Dec in degrees, for the return image.
    size : int or Quantity with angular unit, optional
        If given type is int, then this represents the length of the pixels for
        return image array. if given type is angle Quantity, this represents 
        the size of the image in angular size. The default is 1600.
    pix_scale : float, optional
        The sampling size of the returning image. Users may want to set this
        below unity. The default is 0.262, which corresponds to nanomaggy unit.
    dr : int (9 or 10), optional
        The number of data release of DESI Legacy Survey from which returning
        image will be taken. The default is 10.

    Returns
    -------
    rgbimg : numpy array of shape (N, N, 3)
        The array of RGB color image.
    wcs : WCS axes
        astropy WCS object with three axes 'RA---TAN', 'DEC--TAN', ''.
        
    Examples
    --------
    
    >>> from exgalcosutils.from_legacy_survey import get_lgs_image
    >>> from astropy import unit as u
    >>> from matplotlib import pyplot as plt
    >>> cra, cdec = 128.8913, -1.8492
    >>> rgbimg, wcs = get_lgs_image(cra, cdec, size=1*u.arcmin)
    >>> fig = plt.figure(figsize=(15,15))
    >>> ax = fig.add_subplot(111, projection=wcs, slices=('x','y',0))
    >>> ax.imshow(rgbimg)
    >>> ax.set_xlabel('RA')
    >>> ax.set_ylabel('DEC')

    """
    if type(size) == u.quantity.Quantity:
        SCALE = 25/9*1e-4*u.deg * pix_scale  # scale angle of a pixel
        pix_size = round(((2*size/SCALE).decompose().value))
    elif type(size) == int:
        pix_size = size
    else:
        raise ValueError('`size` should be either type of int '+
                         'astropy Quantity in angular unit')
    try:
        if dr == 9:
            img_query_url = 'https://www.legacysurvey.org/viewer/fits-cutout?'\
            +f'ra={cra:.8f}&dec={cdec:.8f}&width={pix_size}&height={pix_size}'\
                            + f'&layer=ls-dr9&pixscale={pix_scale}&band=grz'

            img_hdu = fits.open(img_query_url)
            wcs = WCS(img_hdu[0].header)

            gimg = img_hdu[0].data[0]
            rimg = img_hdu[0].data[1]
            zimg = img_hdu[0].data[2]
            
            rgbimg = dr2_rgb(rimgs=[gimg, rimg, zimg], bands=['g','r','z'])
            
        elif dr == 10:
            img_query_url = 'https://www.legacysurvey.org/viewer/cutout.fits?'\
            +f'ra={cra:.8f}&dec={cdec:.8f}&width={pix_size}&height={pix_size}'\
                + f'&layer=ls-dr10&pixscale={pix_scale}&bands=griz'

            img_hdu = fits.open(img_query_url)
            h = img_hdu[0].header
            wcs = WCS(h)
            
            if h['BANDS'] == 'grz':
                gimg = img_hdu[0].data[0]
                rimg = img_hdu[0].data[1]
                zimg = img_hdu[0].data[2]
                rgbimg = dr2_rgb(rimgs=[gimg, rimg, zimg], bands=['g','r','z'])
            elif h['BANDS'] == 'griz': 
                gimg = img_hdu[0].data[0]
                rimg = img_hdu[0].data[1]
                iimg = img_hdu[0].data[2]
                zimg = img_hdu[0].data[3]
                rgbimg = dr10_griz_rgb(rimgs=[gimg, rimg, iimg, zimg],
                                       bands=['g','r','i','z'])

    except HTTPError:
        return None, None
    except Exception as e:
        raise e
    return rgbimg, wcs


def get_lgs_image_lupton(cra, cdec, size=1600, pix_scale=1):
    try:
        if type(size) == u.quantity.Quantity:
            SCALE = 25/9*1e-4*u.deg  # scale angle of a pixel
            pix_size = round(((2*size/SCALE).decompose().value))
        elif type(size) == int:
            pix_size = size
        else:
            raise ValueError('`size` should be either type of int '+
                             'astropy Quantity in angular unit')
        
        img_query_url = 'https://www.legacysurvey.org/viewer/fits-cutout?'\
                        + f'ra={cra:.4f}&dec={cdec:.4f}&width={pix_size}&height={pix_size}'\
                            + f'&layer=ls-dr9&pixscale={pix_scale}&band=grz'

        img_hdu = fits.open(img_query_url)

        wcs = WCS(img_hdu[0].header)
        gimg = img_hdu[0].data[0]
        rimg = img_hdu[0].data[1]
        zimg = img_hdu[0].data[2]
        rgbimg = make_lupton_rgb(zimg, rimg, gimg, stretch=0.1)

    except HTTPError:
        return None
    except Exception as e:
        raise e
    return rgbimg, wcs


def get_lgs_galaxies(cra, cdec, ang_limit, get_image=False, **kwargs):
    try:
        half_width = ang_limit.to('deg').value

        ralo, rahi = np.array([-1.,1.])*half_width/np.cos(cdec) + cra
        declo, dechi = np.array([-1.,1.])*half_width + cdec

        photz_url = 'https://www.legacysurvey.org/viewer/photoz-dr9/1/cat.json?'
        cat_bbox = f'ralo={ralo:.4f}&rahi={rahi:.4f}' \
                   + f'&declo={declo:.4f}&dechi={dechi:.4f}'

        photz_query_url = photz_url + cat_bbox
        with conf.set_temp('remote_timeout', 600.0):
            ptab = Table.read(photz_query_url, format='pandas.json')
        if len(ptab)==0:
            return None

        lgs_url = 'https://www.legacysurvey.org/viewer/ls-dr9/cat.fits?'
        query_url = lgs_url + cat_bbox


        with conf.set_temp('remote_timeout', 600.0):
            hdu = fits.open(query_url)

        # _tab_ls = Table.read(query_url)

        if get_image:
            rgbimg, wcs = get_lgs_image(cra, cdec, **kwargs)

        tab_ls = Table([hdu[1].data['ra'], hdu[1].data['dec'],
                        hdu[1].data['type'], hdu[1].data['flux_g'],
                        hdu[1].data['flux_r'], hdu[1].data['flux_z']],
                        names=['ra', 'dec', 'type',
                               'flux_g', 'flux_r', 'flux_z'])

        lco = SkyCoord(ra=tab_ls['ra'], dec=tab_ls['dec'], unit='deg')
        pradec = np.vstack(ptab['rd'])
        pco = SkyCoord(ra=pradec[:,0], dec=pradec[:,1], unit='deg')
        il, ip = match_catalogs(lco, pco, 1.0)

        tab = hstack([tab_ls[il], ptab['phot_z_mean', 'phot_z_std'][ip]])

        # to return a table that comprises galaxies only.
        type_mask = np.logical_not(np.isin(tab['type'], ['PSF', 'DUP']))
        gtab = tab[type_mask]

        # to screen the targets outside the cone with angular radius of ang_limit.
        cco = SkyCoord(ra=cra, dec=cdec, unit='deg')
        gco = SkyCoord(ra=gtab['ra'].data, dec=gtab['dec'].data, unit='deg')

        sep = cco.separation(gco)
        gtab['sep'] = sep
        cone_mask = sep < ang_limit

        if get_image:
            return gtab[cone_mask], rgbimg, wcs
        else:
            return gtab[cone_mask]
    except HTTPError:
        return None
    except Exception as e:
        raise e


def nmgy_to_abmag(f):
    # Pogson relation
    # 1 maggy corresponds to approximately 3361 Jy
    return 22.5 - 2.5*np.log10(f)


def get_info_fig(target, ang_limit, annotate=False, mag_limit=20):
    cra, cdec = target['RAdeg'], target['DEdeg']
    cco = SkyCoord(ra=cra, dec=cdec, unit='deg')
    res = get_lgs_galaxies(cra, cdec, ang_limit, get_image=True)

    if res == None:
        return None
    else:
        gtab, img, wcs = res

        fig = plt.figure(figsize=(15,15))
        ax = fig.add_subplot(111, projection=wcs, slices=('x','y',0))
        ax.imshow(img)

        fov = SphericalCircle((cco.ra, cco.dec), ang_limit.to('deg'),
                                 transform=ax.get_transform('world'),
                                 color='white', fill=False)
        ax.add_patch(fov)
        ax.scatter(cco.ra, cco.dec, marker='+', c='white', s=100,
                   transform=ax.get_transform('world'))
        ax.set_xlabel('RA')
        ax.set_ylabel('DEC')
        ax.set_title(r'$m_{r}<$'+f'{mag_limit:.1f}')
        if annotate:
            gtab.sort(keys=['flux_r','flux_g','flux_z'], reverse=True)
            # TODO annotate only for some bright sources.
            m_r = nmgy_to_abmag(gtab['flux_r'])
            bright = gtab[m_r < mag_limit]

            gco = SkyCoord(ra=bright['ra'], dec=bright['dec'], unit='deg')
            ax.scatter(gco.ra, gco.dec, marker='o', s=100, fc='none', ec='white',
                       transform=ax.get_transform('world'))

            for i in range(len(bright)):
                x, y = gco[i].ra.deg, gco[i].dec.deg
                # ax.scatter(x+0.01,y-0.01,marker=f'${i}$',c='white',
                #             transform=ax.get_transform('world'))
                ax.annotate(str(i), (x,y), (5,-3) ,c='white', fontsize=6,
                            xycoords=ax.get_transform('world'),
                            textcoords='offset pixels', ha='left', va='top')
        # TODO insert title with PSZID

        return gtab


#%% https://github.com/zooniverse/decals/blob/master/decals/a_download_decals/get_images/image_utils.py
# 2018 Mike Walmsley (MIT License)
def decals_internal_rgb(imgs, bands, mnmx=None, arcsinh=None, scales=None,
                        clip=True):
    """
    Given a list of images in the given bands, returns a scaled RGB-like matrix.
    Written by Dustin Lang and used internally by DECALS collaboration
    Copied from https://github.com/legacysurvey/legacypipe/tree/master/py/legacypipe repo
    Not recommended - tends to oversaturate galaxy center
    Args:
        imgs (list): numpy arrays, all the same size, in nanomaggies
        bands (list): strings, eg, ['g','r','z']
        mnmx (min,max), values that will become black/white *after* scaling. ):
        arcsinh (bool): if True, use nonlinear scaling (as in SDSS)
        scales (str): Override preset band scaling. Dict of form {band: (plane index, scale divider)}
        clip (bool): if True, restrict output values to range (0., 1.)
    Returns:
        (np.array) of shape (H, W, 3) of pixel values for colour image
    """

    bands = ''.join(bands)

    grzscales = dict(g=(2, 0.0066),
                     r=(1, 0.01),
                     z=(0, 0.025),
                     )

    if scales is None:
        if bands == 'grz':
            scales = grzscales
        elif bands == 'urz':
            scales = dict(u=(2, 0.0066),
                          r=(1, 0.01),
                          z=(0, 0.025),
                          )
        elif bands == 'gri':
            scales = dict(g=(2, 0.002),
                          r=(1, 0.004),
                          i=(0, 0.005),
                          )
        elif bands == 'ugriz':
            scales = dict(g=(2, 0.0066),
                          r=(1, 0.01),
                          i=(0, 0.05),
                          )
        else:
            scales = grzscales

    h, w = imgs[0].shape
    rgb = np.zeros((h, w, 3), np.float32)
    for im, band in zip(imgs, bands):
        if not band in scales:
            print('Warning: band', band, 'not used in creating RGB image')
            continue
        plane, scale = scales.get(band, (0, 1.))
        rgb[:, :, plane] = (im / scale).astype(np.float32)

    if mnmx is None:
        mn, mx = -3, 10
    else:
        mn, mx = mnmx

    if arcsinh is not None:
        def nlmap(x):
            return np.arcsinh(x * arcsinh) / np.sqrt(arcsinh)

        rgb = nlmap(rgb)
        mn = nlmap(mn)
        mx = nlmap(mx)

    rgb = (rgb - mn) / (mx - mn)

    if clip:
        return np.clip(rgb, 0., 1.)
    return rgb

#%% https://github.com/legacysurvey/imagine/blob/main/map/views.py
# code by Dustin Lang

def sdss_rgb(imgs, bands, scales=None, m=0.02):
    import numpy as np
    rgbscales = {'u': (2,1.5), #1.0,
                 'g': (2,2.5),
                 'r': (1,1.5),
                 'i': (0,1.0),
                 'z': (0,0.4), #0.3
                 }
    if scales is not None:
        rgbscales.update(scales)

    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
        
    # b,g,r = [rimg * rgbscales[b] for rimg,b in zip(imgs, bands)]
    # r = np.maximum(0, r + m)
    # g = np.maximum(0, g + m)
    # b = np.maximum(0, b + m)
    # I = (r+g+b)/3.
    Q = 20
    fI = np.arcsinh(Q * I) / np.sqrt(Q)
    I += (I == 0.) * 1e-6
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        rgb[:,:,plane] = (img * scale + m) * fI / I

    # R = fI * r / I
    # G = fI * g / I
    # B = fI * b / I
    # # maxrgb = reduce(np.maximum, [R,G,B])
    # # J = (maxrgb > 1.)
    # # R[J] = R[J]/maxrgb[J]
    # # G[J] = G[J]/maxrgb[J]
    # # B[J] = B[J]/maxrgb[J]
    # rgb = np.dstack((R,G,B))
    rgb = np.clip(rgb, 0, 1)
    return rgb

def dr2_rgb(rimgs, bands, **ignored):
    return sdss_rgb(rimgs, bands, scales=dict(g=(2,6.0), r=(1,3.4), z=(0,2.2)),
                    m=0.03)

def dr10_griz_rgb(imgs, bands, **kwargs):
    m=0.03
    Q=20
    mnmx=None
    clip=True
    allbands = ['g','r','i','z']
    rgb_stretch_factor = 1.5
    rgbscales=dict(
        g =    (2, 6.0 * rgb_stretch_factor),
        r =    (1, 3.4 * rgb_stretch_factor),
        i =    (0, 3.0 * rgb_stretch_factor),
        z =    (0, 2.2 * rgb_stretch_factor),
        )
    I = 0
    for img,band in zip(imgs, bands):
        plane,scale = rgbscales[band]
        img = np.maximum(0, img * scale + m)
        I = I + img
    I /= len(bands)
    if Q is not None:
        fI = np.arcsinh(Q * I) / np.sqrt(Q)
        I += (I == 0.) * 1e-6
        I = fI / I
    H,W = I.shape
    rgb = np.zeros((H,W,3), np.float32)

    rgbvec = dict(
        g = (0.,   0.,  0.75),
        r = (0.,   0.5, 0.25),
        i = (0.25, 0.5, 0.),
        z = (0.75, 0.,  0.))

    for img,band in zip(imgs, bands):
        _,scale = rgbscales[band]
        rf,gf,bf = rgbvec[band]
        if mnmx is None:
            v = (img * scale + m) * I
        else:
            mn,mx = mnmx
            v = ((img * scale + m) - mn) / (mx - mn)
        if clip:
            v = np.clip(v, 0, 1)
        if rf != 0.:
            rgb[:,:,0] += rf*v
        if gf != 0.:
            rgb[:,:,1] += gf*v
        if bf != 0.:
            rgb[:,:,2] += bf*v
    return rgb
