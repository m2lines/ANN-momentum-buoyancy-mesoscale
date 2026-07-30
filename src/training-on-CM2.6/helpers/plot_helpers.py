import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os
from PIL import Image
from mpl_toolkits.axes_grid1 import make_axes_locatable
import cmocean
import imageio

def create_animation(fun, idx, filename='my-animation.gif', dpi=200, FPS=18, loop=0, deezering=True):
    '''
    See https://pythonprogramming.altervista.org/png-to-gif/
    fun(i) - a function creating one snapshot, has only one input:
        - number of frame i
    idx - range of frames, i in idx
    FPS - frames per second
    filename - animation name
    dpi - set 300 or so to increase quality
    loop - number of repeats of the gif
    '''
    frames = []
    for i in idx:
        fun(i)
        plt.savefig('.frame.png', dpi=dpi, bbox_inches='tight')
        plt.close()
        if deezering:
            frames.append(Image.open('.frame.png').convert('RGB'))
        else:
            frames.append(Image.open('.frame.png'))
        print(f'Frame {i} is created', end='\r')
    os.system('rm .frame.png')
    # How long to persist one frame in milliseconds to have a desired FPS
    duration = 1000 / FPS
    print(f'Animation at FPS={FPS} will last for {len(idx)/FPS} seconds')
    frames[0].save(
        filename, format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=duration,
        loop=loop)
    
def create_animation_ffmpeg(fun, idx, filename='my-video.mp4', dpi=200, FPS=18, resolution=None):
    folder = '.ffmpeg/'+filename.split('.')[0]
    from time import time
    def create_snapshots():
        t0 = time()
        for frame, i in enumerate(idx):
            fun(i)
            plt.savefig(f'{folder}/frame-{frame}.png', dpi=dpi, bbox_inches='tight')
            plt.close()
            nframes = len(idx)
            remaining_frames = nframes - frame
            ETA = (time()-t0) / (frame+1) * remaining_frames
            print(f'Frame {frame}/{nframes} is created, ETA: {ETA}', end='\r')
            
    if os.path.exists(folder):
        if os.path.exists(folder+'/frame-0.png'):
            print(f'Frames already exists in folder {folder}')
            x = input('Do you want to update snapshots?: [y/n]')
            if x=='y':
                create_snapshots()
            elif x=='n':
                print('Frames are not updated\n')
    else:
        os.system(f'mkdir -p {folder}')
        create_snapshots()

    if resolution is None:
        resolution = list(Image.open(f'{folder}/frame-0.png').size)
        for i in [0,1]:
            resolution[i] = (resolution[i]//2)*2
        print(f'Native resolution of snapshots is used: {resolution[0]}x{resolution[1]}\n')
    else:
        for i in [0,1]:
            resolution[i] = (resolution[i]//2)*2
        print(f'Resolution is set to {resolution[0]}x{resolution[1]}\n')

    print(f'Animation {filename} at FPS={FPS} will last for {round(len(idx)/FPS,1)} seconds. The frames are saved to \n{folder}\n')
    ffmpeg_command = f'ffmpeg -y -framerate {FPS} -i {folder}/frame-%d.png -s:v {resolution[0]}x{resolution[1]} -c:v libx264 -profile:v high -crf 20 -pix_fmt yuv420p {filename}'
    print('Running the command:')
    print(f'cd {os.getcwd()}; {ffmpeg_command}')
    try:
        os.system('module load ffmpeg/4.2.4')
    except:
        pass
    try:
        os.system(ffmpeg_command)
    except:
        print('Something went wrong. Try to run the following command in the terminal:\n')
        print('Optional: module load ffmpeg/4.2.4')
        print(f'cd {os.getcwd()}; {ffmpeg_command}')
    
def merge_gifs(gif_files, output_file, fps=20):
    '''
    Note it is purely chatgpt code
    '''
    # Get a list of all GIF files in the input folder

    # Create a list to store individual frames
    frames = []

    # Read each GIF file and extract frames
    for gif_file in gif_files:
        gif_path = os.path.join(gif_file)
        try:
            with imageio.get_reader(gif_path) as reader:
                for frame in reader:
                    frames.append(frame)
        except Exception as e:
            print(f"Error reading {gif_file}: {e}")

    # Write the merged frames to the output GIF
    try:
        with imageio.get_writer(output_file, mode='I', duration=1000//fps, loop=0) as writer:
            for frame in frames:
                writer.append_data(frame)
        print(f"Merged {len(gif_files)} GIFs into {output_file}")
    except Exception as e:
        print(f"Error writing {output_file}: {e}")

def split_gif(input_file, output_folder, n):
    '''
    Note it is purely chatgpt code
    '''
    try:
        with imageio.get_reader(input_file) as reader:
            num_frames = len(reader)
            frames_per_segment = num_frames // n

            if frames_per_segment == 0:
                print("Cannot split into that many segments. Try a smaller value of n.")
                return

            for i in range(n):
                start_frame = i * frames_per_segment
                end_frame = (i + 1) * frames_per_segment if i < n - 1 else num_frames

                os.system('mkdir -p ' + output_folder)
                output_file = os.path.join(output_folder, f"segment_{i}.gif")

                with imageio.get_writer(output_file, mode='I', duration=reader.get_meta_data()['duration'], loop=0) as writer:
                    for frame_number in range(start_frame, end_frame):
                        frame = reader.get_data(frame_number)
                        writer.append_data(frame)

                print(f"Segment {i} saved as {output_file}")

        print(f"Split {input_file} into {n} segments")
    except Exception as e:
        print(f"Error splitting {input_file}: {e}")
    
def default_rcParams(kw={}):
    '''
    Also matplotlib.rcParamsDefault contains the default values,
    but:
    - backend is changed
    - without plotting something as initialization,
    inline does not work
    '''
    plt.plot()
    plt.close()
    rcParams = matplotlib.rcParamsDefault.copy()
    
    # We do not change backend because it can break
    # inlining; Also, 'backend' key is broken and 
    # we cannot use pop method
    for key, val in rcParams.items():
        if key != 'backend':
            rcParams[key] = val

    matplotlib.rcParams.update({
        'font.family': 'MathJax_Main',
        'mathtext.fontset': 'cm',

        'axes.formatter.use_mathtext': True,
        
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.1
    })
    matplotlib.rcParams.update(**kw)

def latex_float(f):
    float_str = "{0:.2g}".format(f)
    if "e" in float_str:
        base, exponent = float_str.split("e")
        return r"{0} \times 10^{{{1}}}".format(base, int(exponent))
    else:
        return float_str
    
def imshow(_q, cbar=True, location='right', cbar_label=None, ax=None, cmap=None, 
    vmax = None, vmin = None, pct=99, axes=False, interpolation='none', normalize='False', normalize_postfix='', **kwargs):

    def rms(x):
        return float(np.sqrt(np.mean(x.astype('float64')**2)))
    def mean(x):
        return float(np.mean(x.astype('float64')))

    if normalize != 'False':
        if normalize == 'mean':
            q_norm = mean(_q)
            q_str = f'$\\mu_x={latex_float(q_norm)}$'
        else:
            q_norm = rms(_q)
            q_str = f'${latex_float(q_norm)}$'    
        q = _q / q_norm
        if len(normalize_postfix) > 0:
            q_str += f' {normalize_postfix}'
    else:
        q = _q

    if q.min() < 0:
        vmax = np.percentile(np.abs(q), pct) if vmax is None else vmax
        vmin = -vmax if vmin is None else vmin
    else:
        vmax = np.percentile(q, pct) if vmax is None else vmax
        vmin = 0 if vmin is None else vmin

    cmap=cmocean.cm.balance if cmap is None else cmap
    
    kw = dict(vmin=vmin, vmax=vmax, cmap=cmap, interpolation=interpolation)
    
    if ax is None:
        ax = plt.gca()

    # flipud because imshow inverts vertical axis
    im = ax.imshow(np.flipud(q), **kw, **kwargs)
    ax.set_xticks([])
    ax.set_yticks([])
    if axes:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
    
    if normalize != 'False':
        ax.text(0.05,0.85,q_str,transform = ax.transAxes, fontsize=8, bbox=dict(boxstyle='round', facecolor='white', alpha=1))

    if cbar:
        divider = make_axes_locatable(ax)
        if location == 'right':
            cax = divider.append_axes('right', size="5%", pad=0.1)
            cbar_kw = dict()
        elif location == 'bottom':
            cax = divider.append_axes('bottom', size="5%", pad=0.1)
            cbar_kw = dict(orientation='horizontal')
        cb = plt.colorbar(im, cax = cax, label=cbar_label, **cbar_kw)
    
    # Return axis to initial image
    plt.sca(ax)
    return im

def set_letters(x=-0.2, y=1.05, fontsize=11, letters=['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p'], color='k'):
    fig = plt.gcf()
    axes = fig.axes
    j = 0
    for ax in axes:
        if hasattr(ax, 'collections'):
            if len(ax.collections) > 0:
                collection = ax.collections[0]
            else:
                collection = ax.collections
            if isinstance(collection, matplotlib.collections.LineCollection):
                print('Colorbar-like object skipped')
            else:
                try:
                    ax.text(x,y,f'({letters[j]})', transform = ax.transAxes, fontweight='bold', fontsize=fontsize, color=color)
                except:
                    print('Cannot set letter', letters[j])
                j += 1
        

def regrid_tripolar(f, glon, glat, dx_deg, loni=None, lati=None):
    '''
    Regrid a curvilinear (tripolar) CM2.6 field to a regular lon/lat grid for cartopy, which
    cannot draw the native grid (geolon is discontinuous near the Arctic and the NH blanks out).

    f            2-D field on the model grid, NaN where masked
    glon, glat   geolon/geolat for the same grid
    dx_deg       nominal grid spacing in degrees, used for the off-domain backstop

    An output cell is drawn only if its NEAREST SOURCE CELL is itself valid. Two traps this
    avoids, both of which produced published-looking but fabricated maps:
      * a plain nearest-neighbour gap fill paints interior holes -- shallow and marginal seas
        that lie below the seafloor at the plotted level -- by extrapolating from whatever wet
        point is closest, and unlike continents these are not hidden by the cartopy LAND feature;
      * a pure DISTANCE threshold cannot fix that, because the masked coastal band is ~2*dx wide
        and so widens with coarsening: any threshold that also scales with dx refills the band,
        and the land imprint then stops growing with resolution.
    The distance cap survives only to kill cells beyond the domain edge entirely.
    '''
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    if loni is None:
        loni = np.arange(-179.5, 180., 1.0)
    if lati is None:
        lati = np.arange(-78.5, 89.5, 1.0)
    LO, LA = np.meshgrid(loni, lati)
    qxy = np.column_stack([LO.ravel() * np.cos(np.deg2rad(LA.ravel())), LA.ravel()])

    lo = ((glon + 180) % 360) - 180
    ok = np.isfinite(f)
    pts = np.column_stack([lo[ok], glat[ok]]); val = f[ok]
    # pad periodically in lon so the interpolation wraps cleanly across +/-180
    pts = np.vstack([pts, pts + [360, 0], pts - [360, 0]]); val = np.concatenate([val, val, val])
    lin = griddata(pts, val, (LO, LA), method='linear')
    nn = griddata(pts, val, (LO, LA), method='nearest')
    out = np.where(np.isfinite(lin), lin, nn)

    ap = np.column_stack([lo.ravel(), glat.ravel()]); av = ok.ravel().astype(float)
    fin = np.isfinite(ap[:, 0]) & np.isfinite(ap[:, 1]); ap, av = ap[fin], av[fin]
    ap = np.vstack([ap, ap + [360, 0], ap - [360, 0]]); av = np.concatenate([av, av, av])
    axy = np.column_stack([ap[:, 0] * np.cos(np.deg2rad(ap[:, 1])), ap[:, 1]])
    dn, idx = cKDTree(axy).query(qxy)
    keep = (av[idx] > 0.5) & (dn <= 1.5 * dx_deg)
    return loni, lati, np.where(keep.reshape(LO.shape), out, np.nan)


def native_lonlat(f, glon, glat):
    '''
    Prepare a curvilinear CM2.6 field for cartopy pcolormesh WITHOUT interpolating, the way the
    upstream momentum-paper notebooks plot (notebooks/Figure-1, Figure-3). Preferred over
    regrid_tripolar: each model cell is drawn as its own quad, so the plotted mask is exactly the
    data's NaN mask and no value can be fabricated.

    Two fixes are needed for CM2.6 specifically:
      * geolon spans -279.7..+79.7, outside the [-180,180] that ccrs.PlateCarree expects, so the
        Pacific is silently dropped. Normalise it. (This -- not the tripolar fold -- is why naive
        native plotting appears to "not work" on this grid.)
      * after normalising, ~330 cells straddle the +/-180 seam; pcolormesh would smear those across
        the whole map, so blank them. They are a seam artifact, not data.

    Returns (lon, lat, field) ready for ax.pcolormesh(..., transform=ccrs.PlateCarree()).
    '''
    lon = ((np.asarray(glon) + 180) % 360) - 180
    lat = np.asarray(glat)
    out = np.array(f, dtype='float64', copy=True)
    bad = np.zeros(lon.shape, dtype=bool)
    for arr, axis in ((np.abs(np.diff(lon, axis=1)), 1), (np.abs(np.diff(lon, axis=0)), 0)):
        big = arr > 180.
        if axis == 1:
            bad[:, :-1] |= big; bad[:, 1:] |= big
        else:
            bad[:-1, :] |= big; bad[1:, :] |= big
    out[bad] = np.nan
    return lon, lat, out
