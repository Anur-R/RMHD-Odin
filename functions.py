import os 
import yt 
import numpy as np
from natsort import natsorted

def file_structure(data_dir):
    # View the data directory, return the immediate subdirectories
    data_subdir = [name.path for name in os.scandir(data_dir) if name.is_dir()]
    return natsorted(data_subdir)

def load_snapshot(data_dir, fields):
    # load a snapshot and pull the necessary fields 
    # into a covering grid. 
    ds = yt.load(data_dir)

    max_level = ds.index.max_level
    ref = int(np.prod(ds.ref_factors[0:ds.index.max_level]))
    low = ds.domain_left_edge
    dims = ds.domain_dimensions*ref
    
    cube = ds.covering_grid(max_level, left_edge=low, dims=dims, fields=fields)
    
    return ds, cube

def lorentz(cube):  
    gvx = cube["WVX"].d
    gvy = cube["WVY"].d
    gvz = cube["WVZ"].d
    nx, ny, nz = gvx.shape

    lorentz = np.zeros_like(gvx)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                lorentz[i, j, k] = np.sqrt(1 + gvx[i, j, k] ** 2 + gvy[i, j, k] ** 2 + gvz[i, j, k] ** 2)

    return lorentz

def velocities(cube): 
    gvx = cube["WVX"].d
    gvy = cube["WVY"].d
    gvz = cube["WVZ"].d
    l_factor = lorentz(cube)
    return gvx/l_factor, gvy/l_factor, gvz/l_factor

def make_ke_ps(ds, cube): 
    # take a snapshot and make a power spectrum 
    # for the kinetic energy
    
    rho = cube["RHOB"].d
    vx, vy, vz = velocities(cube)
    nx, ny, nz = rho.shape
    
    nindex_rho = 1/3
    
    Kk = np.zeros( (nx//2+1, ny//2+1, nz//2+1))

    for vel in [vx, vy, vz]: 
        ru = np.fft.fftn(rho**nindex_rho * vel)[0:nx//2+1, 0:ny//2+1, 0:nz//2+1] 
        ru = 8 * ru/(nx*ny*nz)
        Kk += 0.5 * np.abs(ru)**2
        
    L = (ds.domain_right_edge - ds.domain_left_edge).d 

    kx = np.fft.rfftfreq(nx) * nx / L[0]
    ky = np.fft.rfftfreq(ny) * ny / L[1]
    kz = np.fft.rfftfreq(nz) * nz / L[2]

    kmin = np.min(1/L) 
    kmax = np.min(0.5 * ds.domain_dimensions/L) # for 3D only
    #kmax = 128 ### For 2D only

    kbins = np.arange(kmin, kmax, kmin) 
    N = len(kbins)

    kx3d, ky3d, kz3d = np.meshgrid(kx, ky, kz, indexing="ij")
    k = np.sqrt(kx3d**2 + ky3d**2, kz3d**2)

    whichbin = np.digitize(k.flat, kbins)
    ncount = np.bincount(whichbin)

    E_spectrum = np.zeros(len(ncount) - 1) 

    for n in range(1, len(ncount)): 
        E_spectrum[n-1] = np.sum(Kk.flat[whichbin == n])

    k = 0.5 * (kbins[0:N-1] + kbins[1:N])
    E_spectrum = E_spectrum[1:N]

    return k, E_spectrum

def ke_ps_for_each_snapshot(data_dir):
    # take a simulation and load the power 
    # spectrum for each snapshot into a 
    # 3D matrix. There is probably a better 
    # way to store the data but I don't care.
    subdir = file_structure(data_dir)

    fields = ["RHOB", "WVX", "WVY", "WVZ"]

    data = np.zeros((2, 126, len(subdir)))

    for i in range(len(subdir)):
        ds, cube = load_snapshot(subdir[i], fields)
        data[:,:, i] = make_ke_ps(ds, cube)

    return data, subdir

def make_vel_field(ds, cube):
    vx, vy, vz = velocities(cube)
    dims = ds.domain_dimensions 

    velocity_field = np.zeros( (*dims, 3) ) 

    for i in range(dims[0]):
        for j in range(dims[1]): 
            for k in range(dims[2]): 
                velocity_field[i, j, k, :] = vx[i, j, k], vy[i, j, k], vz[i, j, k]

    return dims, velocity_field

def vel_structure_func(ds, cube, order, nbins): 
    dims, velocity_field = make_vel_field(ds, cube) 
    
    # create list of grid points to iterate through
    ranges = [np.arange(0, dim) for dim in dims]
    grid_points = np.array(np.meshgrid(*ranges)).T.reshape(-1, 3)

    # some values to keep track of
    steps = np.prod(dims)

    # total number of iterations to go through
    iterations = int(steps * (steps-1)/2)
    distances = np.zeros(iterations)
    vel_structure = np.zeros(iterations)

    count = 0
    for i in range(steps-1):  
        # take the first coordinate
        p1 = grid_points[i]
        for p2 in grid_points[i+1:]: 
            # for every other coordinate find distance 
            distances[count] = np.linalg.norm(p2 - p1)

            # calculate velocity structure between the points
            vel_structure[count] = ( np.linalg.norm(velocity_field[*p1, :] - velocity_field[*p2, :]) )**order

            count += 1

    # create bins to calculate the averages
    bins = np.linspace(np.min(distances), np.max(distances), nbins+1) 

    ensamble_avg = np.zeros(nbins)

    for i in range(nbins): 
        # create mask where distances fall below a specific bin
        # use mask on the vel_structure and average them
        ensamble_avg[i] = np.average(vel_structure[np.where(distances < bins[i+1])])
    return bins[:nbins], ensamble_avg

def main(): 
    #gang shit
    return

if __name__ == "__main__": 
    main()
