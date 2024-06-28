import os 
import yt 
import numpy as np

def file_structure(data_dir):
    # View the data directory, return the names of the snapshots
    data_subdir = [name for name in os.listdir(data_dir)] 
    data_subdir.sort(key=str.lower) 
    return data_subdir

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

def make_ke_ps(snapshot): 
    # take a snapshot and make a power spectrum 
    # for the kinetic energy
    ke_vec = ["RHOB", "WVX", "WVY", "WVZ"]
    ds, cube = load_snapshot(snapshot, ke_vec)
    
    rho = cube["RHOB"].d
    gvx = cube["WVX"].d
    gvy = cube["WVY"].d
    gvz = cube["WVZ"].d
    
    nx, ny, nz = rho.shape

    lorentz = np.zeros_like(rho)
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                lorentz[i, j, k] = np.sqrt(1 + gvx[i, j, k] ** 2 + gvx[i, j, k] ** 2 + gvz[i, j, k] ** 2)
                
    vx = gvx/lorentz  
    vy = gvy/lorentz  
    vz = gvz/lorentz 
    
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
    #kmax = np.min(0.5 * dims/L)
    kmax = 128 ### For 2D only, please delete line and use line above

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

    data = np.zeros((2, 254, len(subdir)))

    for i in range(len(subdir)):
        path = data_dir + subdir[i]
        data[:,:, i] = make_ke_ps(path)

    return data, subdir

def main(): 
    #gang shit
    return

if __name__ == "__main__": 
    main()
