import numpy as np
import h5py
import os,sys
import pickle
# !import code; code.interact(local=vars())                                                          
# pdb.set_trace = lambda: None 

def recolor(seg, numC=128):
    out = (seg%numC).astype(np.uint8)+1
    out[seg==0] = 0
    return out

def label_chunk_v2(seg0, fns, numC, rr=1, rm_sz=0, m_type=np.uint64, num_conn=8):
    # label chunks or slices
    from skimage.measure import label
    sz = fns(seg0, 0).shape
    numD = len(sz)
    
    mid = 0
    seg = [None]*numC
    for zi in range(numC):
        print('%d/%d [%d], '%(zi,numC,mid)),
        sys.stdout.flush()
        seg_c = label(fns(seg0, zi)>0).astype(m_type)

        if numD==2:
            seg_c = seg_c[np.newaxis]
        if rm_sz>0:
            seg_c = remove_small(seg_c, rm_sz)
            seg_c = seg_c[:,::rr,::rr]
            # preserve continuous id
            seg_c = relabel(seg_c).astype(m_type)

        if zi == 0: # first seg, relabel seg index        
            seg[zi] = seg_c
            mid += seg[zi].max()
            rlA = np.arange(mid+1,dtype=m_type)
        else: # link to previous slice
            slice_b = seg[zi-1][-1]
            slice_t = seg_c[0]            
            slices = label(np.stack([slice_b>0, slice_t>0],axis=0), num_conn).astype(m_type)

            # create mapping for seg cur
            lc = np.unique(seg_c);lc=lc[lc>0]
            rl_c = np.zeros(int(lc.max()+1), dtype=int)
            # merge curr seg
            # for 1 pre seg id -> slices id -> cur seg ids
            l0_p = np.unique(slice_b*(slices[0]>0))
            for l in l0_p:
                sid = np.unique(slices[0]*(slice_b==l))
                sid = sid[sid>0]
                cid = np.unique(slice_t*np.in1d(slices[1].reshape(-1),sid).reshape(sz[-2:]))
                rl_c[cid[cid>0]] = l
            
            # new id
            new_num = np.where(rl_c==0)[0][1:] # except the first one
            new_id = np.arange(mid+1,mid+1+len(new_num),dtype=m_type)
            rl_c[new_num] = new_id            
            seg[zi] = rl_c[seg_c]
            mid += len(new_num)
            
            # update global id
            rlA = np.hstack([rlA,new_id])
            # merge prev seg
            # for 1 cur seg id -> slices id -> prev seg ids
            l1_c = np.unique(slice_t*(slices[1]>0))
            for l in l1_c:
                sid = np.unique(slices[1]*(slice_t==l))
                sid = sid[sid>0]
                pid = np.unique(slice_b*np.in1d(slices[0].reshape(-1),sid).reshape(sz[-2:]))
                pid = pid[pid>0]
                # get all previous m-to-1 labels
                pid_p = np.where(np.in1d(rlA,rlA[pid]))[0]
                if len(pid_p)>1:
                    rlA[pid_p] = pid.max()
        # memory reduction: each seg
        m2_type = getSegType(seg[zi].max())
        seg[zi] = seg[zi].astype(m2_type)
        print(m2_type),
    # memory reduction: final output
    m2_type = getSegType(rlA.max())
    rlA = rlA.astype(m2_type)
    print('output type:',m2_type)

    return rlA[np.vstack(seg)]


def label_chunk(fns, numC, rr=1, rm_sz=0):
    # label chunks or slices
    from skimage.measure import label

    sz = fns(0).shape
    numD = len(sz)
    if numD==3: # first dim is z
        sz=sz[1:]
    mid = 0
    seg = [None]*numC
    for zi in range(numC):
        im = label(fns(zi)>0)
        if rm_sz>0:
            im = remove_small(im, rm_sz)
            if numD == 3:
                im = im[:,::rr,::rr]
            else:
                im = im[::rr,::rr]
        if zi == 0: # first seg, relabel seg index        
            seg[zi] = relabel(im)
            mid += seg[zi].max()
            rlA = np.arange(mid+1,dtype=np.uint32)
        else: # link to previous slice
            if numD==3:
                seg_p = seg[zi-1][-1]
            else:
                seg_p = seg[zi-1]
            print(seg_p.shape,im.shape)
            im2 = label(np.stack([seg_p>0, im>0],axis=0))                
            l2 = np.unique(im2);l2=l2[l2>0]
            l0 = np.unique(im2[0]);l0=l0[l0>0]
            l1 = l2[np.in1d(l2,l0,invert=True)]
            
            # relabel array
            rl = np.zeros(l2.max()+1,dtype=int)
            rlA = np.hstack([rlA,np.arange(mid+1,mid+len(l1)+1,dtype=np.uint32)])
            for l in l0:
                pid = np.unique(seg_p*(im2[0]==l))
                pid = pid[pid>0]
                rl[l] = pid.max()
                rlA[pid] = pid.max()
            for li,l in enumerate(l1):
                rl[l] = li+mid+1
            seg[zi] = rl[im2[1:]]
            mid += len(l1)
    if numD == 3:
        return rlA[np.vstack(seg)]
    else:
        return rlA[np.stack(seg,axis=0)]


def removeSeg(seg, did):
    sz = seg.shape 
    seg = seg.reshape(-1)
    seg[np.in1d(seg,did)] = 0
    seg = seg.reshape(sz)

def remove_small(seg, thres=100,bid=None,nid=0,invert=False,do_bb=True):
    if do_bb:
        bb= get_bb(seg>0)
        ndim = seg.ndim
        if ndim==2:
            seg_b = seg[bb[0]:bb[1]+1,bb[2]:bb[3]+1]
        elif ndim==3:
            seg_b = seg[bb[0]:bb[1]+1,bb[2]:bb[3]+1,bb[4]:bb[5]+1]
    else:
        seg_b = seg
    sz = seg_b.shape

    if bid is None:
        uid, uc = np.unique(seg_b, return_counts=True)
        bid = uid[uc<thres]
    
    # new variable/copy mem
    seg_b = seg_b.reshape(-1)
    seg_b[np.in1d(seg_b,bid,invert=invert)] = nid

    if do_bb:
        seg[bb[0]:bb[1]+1,bb[2]:bb[3]+1,bb[4]:bb[5]+1] = seg_b.reshape(sz) 
    else:
        seg = seg_b.reshape(sz)

    return seg

def seg_assign(seg, segM, bid,nid=0,invert=False,do_bb=True):
    if do_bb:
        bb= get_bb(segM>0)
        ndim = seg_M.ndim
        if ndim==2:
            seg_b = seg[bb[0]:bb[1]+1,bb[2]:bb[3]+1]
            segM_b = segM[bb[0]:bb[1]+1,bb[2]:bb[3]+1]
        elif ndim==3:
            seg_b = seg[bb[0]:bb[1]+1,bb[2]:bb[3]+1,bb[4]:bb[5]+1]
            segM_b = segM[bb[0]:bb[1]+1,bb[2]:bb[3]+1,bb[4]:bb[5]+1]
    else:
        seg_b = seg
        segM_b = segM

    sz = seg_b.shape

    # new variable/copy mem
    seg_b = seg_b.reshape(-1)
    segM_b = segM_b.reshape(-1)
    seg_b[np.in1d(segM_b,bid,invert=invert)] = nid

    if do_bb:
        seg[bb[0]:bb[1]+1,bb[2]:bb[3]+1,bb[4]:bb[5]+1] = seg_b.reshape(sz) 
    else:
        seg = seg_b.reshape(sz)
    return seg



def seg2Vast(seg):                                                                                   
    # convert to 24 bits                                                                             
    return np.stack([seg//65536, seg//256, seg%256],axis=2).astype(np.uint8)  

def vast2Seg(seg):
    # convert to 24 bits
    if seg[:,:,0].max()>0: # 8big
        return seg[:,:,0].astype(np.uint8)
    else:
        return seg[:,:,0].astype(np.uint16)*65536+seg[:,:,1].astype(np.uint16)*256+seg[:,:,2].astype(np.uint16)

def readVastSeg(fn):
    a= open(fn).readlines()
    # remove comments
    st_id = 0
    while a[st_id][0] in ['%','\\']:
        st_id+=1
    # remove segment name
    out = np.zeros((len(a)-st_id-1,24), dtype=int)
    name = [None]*(len(a)-st_id-1)
    for i in range(st_id+1,len(a)):
        out[i-st_id-1] = np.array([int(x) for x in a[i][:a[i].find('"')].split(' ') if len(x)>0])
        name[i-st_id-1] = a[i][a[i].find('"')+1:a[i].rfind('"')]
    return out,name

def getSegType(mid):
    m_type = np.uint64
    if mid<2**8:
        m_type = np.uint8
    elif mid<2**16:
        m_type = np.uint16
    elif mid<2**32:
        m_type = np.uint32
    return m_type

def relabelType(seg):
    mid = seg.max()+1
    m_type = getSegType(mid)
    return seg.astype(m_type)

def relabel(seg, uid=None,nid=None,do_sort=False,do_type=False):
    if do_sort:
        uid,_ = seg2Count(seg,do_sort=True)
    else:
        # get the unique labels
        if uid is None:
            uid = np.unique(seg)
        else:
            uid = np.array(uid)
    if (uid>0).sum()>0:
        uid = uid[uid>0] # leave 0 as 0, the background seg-id
        # get the maximum label for the segment
        mid = int(uid.max()) + 1

        # create an array from original segment id to reduced id
        # format opt
        m_type = seg.dtype
        if do_type:
            mid2 = len(uid) if nid is None else nid.max()+1
            m_type = getSegType(mid2)

        mapping = np.zeros(mid, dtype=m_type)
        if nid is None:
            mapping[uid] = np.arange(1,1+len(uid), dtype=m_type)
        else:
            mapping[uid] = nid.astype(m_type)
        seg[seg>=mid] = 0     
        return mapping[seg]
    else:
        return seg

def seg2Count(seg,do_sort=True):
    if seg.max()>1:
        segIds,segCounts = np.unique(seg,return_counts=True)
        if do_sort:
            sort_id = np.argsort(-segCounts)
            segIds=segIds[sort_id]
            segCounts=segCounts[sort_id]
    else:
        segIds=[1];segCounts=[np.count_nonzero(seg)]
    return segIds, segCounts

def seg2largest(seg,numF=8):
    from skimage.measure import label
    seg = label(seg,numF)
    if seg.max()>1:
        segIds,segCounts = np.unique(seg,return_counts=True)
        segIds=segIds[1:];segCounts=segCounts[1:]
        seg = (seg==segIds[np.argmax(segCounts)]).astype(np.uint8)
    return seg


def writepkl(filename, content):
    with open(filename, "wb") as f:
        if isinstance(content, (list,)):
            for val in content:
                pickle.dump(val, f)
        else:
            pickle.dump(content, f)

def readpkl(filename):
    data = []
    with open(filename, "rb") as f:
        while True:
            try:
                data.append(pickle.load(f))
            except:
                break
    return data

def readtxt(filename):
    a= open(filename)
    content = a.readlines()
    a.close()
    return content

def U_mkdir(fn):
    if not os.path.exists(fn):
        os.makedirs(fn)

def writetxt(filename, content):
    a= open(filename,'w')
    if isinstance(content, (list,)):
        for ll in content:
            if '\n' not in ll:
                ll += '\n'
            a.write(ll)
    else:
        a.write(content)
    a.close()

def getCellBbox(cid, dilate=[2,5,5]):                                                                
    sz128 = [773,832,832]                                                                            
    # 120x128x128nm                                                                                  
    D0='/mnt/coxfs01/donglai/data/JWR/snow_cell/'
    bb0=np.loadtxt(D0+'cell_ind/daniel/cell_daniel_bb128nm.txt').astype(int)[cid-1,1:-1]             
    # dilate Daniel's bounding box                                                                   
    if len(dilate)>0:                                                                                
        bb0[::2] = bb0[::2] - dilate                                                                 
        bb0[1::2] = bb0[1::2] + dilate                                                               
        bb0[bb0<0] = 0                                                                               
        bb0[1] = min(bb0[1], sz128[0])                                                               
        bb0[3] = min(bb0[3], sz128[1])                                                               
        bb0[5] = min(bb0[5], sz128[2])                                                               
                                                                                                     
    # 30x8x8nm                                                                                       
    bbP = np.array([4,4,16,16,16,16])*bb0                                                            
    # add z-pad and missing first slice                                                              
    bbP = bbP+[8,8,-512,-512,-512,-512]                                                              
    bbP[bbP<0]=0                                                                                     
    return bb0,bbP   

def readh5_b(filename, sz, datasetname='main'):
    import h5py
    tmp = np.unpackbits(np.array(h5py.File(filename,'r')[datasetname]))
    return tmp[:np.prod(sz)].reshape(sz)


def readh5(filename, datasetname='main', rr=[1,1,1]):
    fid=h5py.File(filename,'r')
    if isinstance(datasetname, (list,)):
        out = [None] *len(datasetname)
        for i,dd in enumerate(datasetname):
            sz = len(fid[dd].shape)
            if sz==1:
                out[i] = np.array(fid[dd][::rr[0]])
            elif sz==2:
                out[i] = np.array(fid[dd][::rr[0],::rr[1]])
            elif sz==3:
                out[i] = np.array(fid[dd][::rr[0],::rr[1],::rr[2]])
            else:
                out[i] = np.array(fid[dd])
    else:                                                                                            
        sz = len(fid[datasetname].shape)
        if sz==1:
            out = np.array(fid[datasetname][::rr[0]])
        elif sz==2:
            out = np.array(fid[datasetname][::rr[0],::rr[1]])
        elif sz==3:
            out = np.array(fid[datasetname][::rr[0],::rr[1],::rr[2]])
        else:
            out = np.array(fid[datasetname])
    return out

def writeh5(filename, dtarray, datasetname='main'):
    fid=h5py.File(filename,'w')                                                                      
    if isinstance(datasetname, (list,)):                                                             
        for i,dd in enumerate(datasetname):                                                          
            ds = fid.create_dataset(dd, dtarray[i].shape, compression="gzip", dtype=dtarray[i].dtype)
            ds[:] = dtarray[i]                                                                       
    else:                                                                                            
        ds = fid.create_dataset(datasetname, dtarray.shape, compression="gzip", dtype=dtarray.dtype) 
        ds[:] = dtarray                                                                              
    fid.close()    

def list_create(chunk):
    if len(chunk)==1:
        out = [None for zi in range(chunk[0])]
    elif len(chunk)==2:
        out = [[None for yi in range(chunk[1])] for zi in range(chunk[0])]
    elif len(chunk)==3:
        out = [[[None for xi in range(chunk[2])] for yi in range(chunk[1])] for zi in range(chunk[0])]
    return out

def bfly_h5(h5Name, x0, x1, y0, y1, z0, z1, zyx_sz, zyx0, dt=np.uint16):
    import h5py
    result = np.zeros((z1-z0, y1-y0, x1-x0), dt)
    c0 = max(0,x0-zyx0[2]) // zyx_sz[2] # floor
    c1 = (x1-zyx0[2]+zyx_sz[2]-1) // zyx_sz[2] # ceil
    r0 = max(0,y0-zyx0[1]) // zyx_sz[1]
    r1 = (y1-zyx0[1]+zyx_sz[1]-1) // zyx_sz[1]
    d0 = max(0,z0-zyx0[0]) // zyx_sz[0]
    d1 = (z1-zyx0[0]+zyx_sz[0]-1) // zyx_sz[0]
    zyx_num = np.prod(zyx_sz)
  
    mid = 0
    for zid in range(d0, d1):
        for yid in range(r0, r1):
            for xid in range(c0, c1):
                path = h5Name(xid, yid, zid)
                """
                path = h5Name % (xid*zyx_sz[2]+zyx0[2],\
                                 yid*zyx_sz[1]+zyx0[1],\
                                 zid*zyx_sz[0]+zyx0[0])
                """
                if os.path.exists(path): 
                    fid = h5py.File(path,'r')['main']
                    xp0 = xid * zyx_sz[2] + zyx0[2]
                    xp1 = (xid+1) * zyx_sz[2]+ zyx0[2]
                    yp0 = yid * zyx_sz[1] + zyx0[1]
                    yp1 = (yid + 1) * zyx_sz[1]+ zyx0[1]
                    zp0 = zid * zyx_sz[0] + zyx0[0]
                    zp1 = (zid + 1) * zyx_sz[0]+ zyx0[0]

                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    z0a = max(z0, zp0)
                    z1a = min(z1, zp1)
                    if dt==np.bool_:
                        tmp = np.unpackbits(np.array(fid))[:zyx_num].reshape(zyx_sz)[z0a-zp0:z1a-zp0,y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]
                    else:
                        if fid.ndim==4:
                            tmp = np.array(fid[:,z0a-zp0:z1a-zp0,y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0])
                        elif fid.ndim==3:
                            tmp = np.array(fid[z0a-zp0:z1a-zp0,y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0])
                    if tmp is not None and tmp.max()>0:
                        result[z0a-z0:z1a-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = mid+relabel(tmp)
                        mid = tmp.max()
    return result


# get one bbox
def bfly_bbox(ff, x0, x1, y0, y1, z0, z1, tile_sz, dim4=-1,dt=np.uint8):
    result = np.zeros((z1-z0, y1-y0, x1-x0), dt)
    c0 = x0 // tile_sz[2] # floor
    c1 = (x1 + tile_sz[2]-1) // tile_sz[2] # ceil
    r0 = y0 // tile_sz[1]
    r1 = (y1 + tile_sz[1]-1) // tile_sz[1]
    d0 = z0 // tile_sz[0]
    d1 = (z1 + tile_sz[0]-1) // tile_sz[0]
    #print 'bfly: ',d0,d1,r0,r1,c0,c1
    for depth in range(d0, d1):
        for row in range(r0, r1):
            for column in range(c0, c1):
                patch = ff[depth][row][column]
                xp0 = column * tile_sz[2]
                xp1 = (column+1) * tile_sz[2]
                yp0 = row * tile_sz[1]
                yp1 = (row + 1) * tile_sz[1]
                zp0 = depth * tile_sz[0]
                zp1 = (depth + 1) * tile_sz[0]
                if patch is not None:
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    z0a = max(z0, zp0)
                    z1a = min(z1, zp1)
                    if dim4==-1:
                        result[z0a-z0:z1a-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = \
                                np.array(patch[z0a-zp0:z1a-zp0, y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0])
                    else:
                        result[z0a-z0:z1a-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = \
                                np.array(patch[dim4,z0a-zp0:z1a-zp0, y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0])
    return result


def bflyWhole(fns, x0, x1, y0, y1, z0, z1, dt=np.uint8, tile_ratio=1, tile_resize_mode='bilinear', do_trans=False):
    import scipy
    # [row,col]
    # no padding at the boundary
    # st: starting index 0 or 1
    result = np.zeros((z1-z0, y1-y0, x1-x0), dt)
    for z in range(z0, z1):
        path = fns[z]
        if not os.path.exists(path):
            patch = None
            # print('not exist:',path)
            # patch = 128*np.ones((tile_sz,tile_sz),dtype=np.uint8)
        else:
            if path[-3:]=='tif':
                import tifffile
                patch = tifffile.imread(path)
            else:
                patch = scipy.misc.imread(path, 'L')
        if patch is not None:
            if do_trans:
                patch = patch.transpose((1,0))
            if tile_ratio != 1:
                patch = scipy.misc.imresize(patch, tile_ratio*100, tile_resize_mode)
            result[z-z0] = patch[y0:y1, x0:x1]
    return result

def bfly_v2(fns, x0, x1, y0, y1, z0, z1, tile_sz, dt=np.uint8,tile_st=[1,1], tile_ratio=1, tile_resize_mode='bilinear'):
    import scipy
    if not isinstance(tile_sz, (list,)):
        tile_sz = [tile_sz, tile_sz]
    # [row,col]
    # no padding at the boundary
    # st: starting index 0 or 1
    result = np.zeros((z1-z0, y1-y0, x1-x0), dt)
    c0 = x0 // tile_sz[1] # floor
    c1 = (x1 + tile_sz[1]-1) // tile_sz[1] # ceil
    r0 = y0 // tile_sz[0]
    r1 = (y1 + tile_sz[0]-1) // tile_sz[0]
    for z in range(z0, z1):
        pattern = fns[z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                path = pattern%(row+tile_st[0], column+tile_st[1])
                #print(path)
                #import pdb; pdb.set_trace()
                if not os.path.exists(path):
                    patch = None
                    #print('not exist:',path)
                    # patch = 128*np.ones(tile_sz,dtype=np.uint8)
                else:
                    #print('load:',path)
                    try:
                        if path[-3:]=='tif':
                            import tifffile
                            patch = tifffile.imread(path)
                        else:
                            patch = scipy.misc.imread(path, 'L')
                        if tile_ratio != 1:
                            patch = scipy.misc.imresize(patch, tile_ratio, tile_resize_mode)
                    except:
                        patch = 128*np.ones(tile_sz,dtype=np.uint8)
                if patch is not None:
                    # exception: last tile may not have the right size
                    psz = patch.shape
                    xp0 = column * tile_sz[1]
                    xp1 = min(xp0+psz[1], (column+1)*tile_sz[1])

                    yp0 = row * tile_sz[0]
                    yp1 = min(yp0+psz[0], (row+1)*tile_sz[0])

                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    try:
                        result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]
                    except:
                        import pdb; pdb.set_trace()
    return result


def bfly(fns, x0, x1, y0, y1, z0, z1, tile_sz, dt=np.uint8,tile_st=[1,1], tile_ratio=1, tile_resize_mode='bilinear'):
    from scipy.misc import imread
    if not isinstance(tile_sz, (list,)):
        tile_sz = [tile_sz, tile_sz]
    # [row,col]
    # no padding at the boundary
    # st: starting index 0 or 1
    result = np.zeros((z1-z0, y1-y0, x1-x0), dt)
    c0 = x0 // tile_sz[1] # floor
    c1 = (x1 + tile_sz[1]-1) // tile_sz[1] # ceil
    r0 = y0 // tile_sz[0]
    r1 = (y1 + tile_sz[0]-1) // tile_sz[0]
    for z in range(z0, z1):
        pattern = fns[z]
        for row in range(r0, r1):
            for column in range(c0, c1):
                path = pattern.format(row=row+tile_st[0], column=column+tile_st[1])
                if not os.path.exists(path):
                    patch = None
                    # print('not exist:',path)
                    # patch = 128*np.ones(tile_sz,dtype=np.uint8)
                else:
                    if path[-3:]=='tif':
                        import tifffile
                        patch = tifffile.imread(path)
                    else:
                        patch = imread(path, 'L')
                xp0 = column * tile_sz[1]
                xp1 = (column+1) * tile_sz[1]
                yp0 = row * tile_sz[0]
                yp1 = (row + 1) * tile_sz[0]
                if patch is not None:
                    if tile_ratio != 1:
                        patch = scipy.misc.imresize(patch, tile_ratio*100, tile_resize_mode)
                    x0a = max(x0, xp0)
                    x1a = min(x1, xp1)
                    y0a = max(y0, yp0)
                    y1a = min(y1, yp1)
                    result[z-z0, y0a-y0:y1a-y0, x0a-x0:x1a-x0] = patch[y0a-yp0:y1a-yp0, x0a-xp0:x1a-xp0]
    return result


def folderV2Seg(Do,dt=np.uint16,maxF=-1,rr=[1,1,1],fns=None):
    from scipy.misc import imread
    import glob
    if fns is None:
        fns = sorted(glob.glob(Do+'*.png'))
    numF = len(fns)
    if maxF>0: 
        numF = min(numF,maxF)

    numF = numF//rr[0]
    for zi in range(numF):
        if os.path.exists(fns[zi*rr[0]]):
            sz = np.array(imread(fns[zi*rr[0]]).shape)[:2]//rr[1:]
            break

    seg = np.zeros((numF,sz[0],sz[1]), dtype=dt)
    if dt==np.uint8:
        for zi in range(numF):
            if os.path.exists(fns[zi*rr[0]]):
                tmp = imread(fns[zi*rr[0]])
                if len(tmp.shape)==3:
                    tmp = tmp[:,:,0]
                seg[zi] = tmp[::rr[1],::rr[2]] 
    else:
        for zi in range(numF):
            if os.path.exists(fns[zi*rr[0]]):
                seg[zi] = vast2Seg(imread(fns[zi*rr[0]]))[::rr[1],::rr[2]]
    return seg


def ngLayer(data,res,oo=[0,0,0],tt='segmentation'):
    # tt: image or segmentation
    import neuroglancer
    return neuroglancer.LocalVolume(data,volume_type=tt,voxel_size=res,offset=oo)

def visLayer(viewer, data, name, res):
    with viewer.txn() as s:
        s.layers.append(name=name, layer=ngLayer(data,res))

def get_union(a,b):                                                                                  
    #[xmin,xmax,ymin,ymax]                                                                           
    ll=len(a)                                                                                        
    out=[None]*ll                                                                                    
    for i in range(0,ll,2):
        out[i] = min(a[i],b[i])
    for i in range(1,ll,2):                                                                          
        out[i] = max(a[i],b[i])                                                                      
    return out      

def get_bb(seg, do_count=False):
    dim = len(seg.shape)
    a=np.where(seg>0)
    if len(a)==0:
        return [-1]*dim*2
    out=[]
    for i in range(dim):
        out+=[a[i].min(), a[i].max()]
    if do_count:
        out+=[len(a[0])]
    return out

def get_bb_label(seg, do_count=False, uid=None):
    dim = len(seg.shape)
    if uid is None:
        uid = np.unique(seg)
        uid = uid[uid>0]
    out = np.zeros((len(uid),dim*2+1+do_count),dtype=np.uint32)
    #print('#bbox: ',len(uid))
    for i,j in enumerate(uid):
        out[i,0] = j
        a=np.where(seg==j)
        if len(a[0])>0:
            for k in range(dim):
                out[i,1+k*2:3+k*2] = [a[k].min(), a[k].max()]
            if do_count:
                out[i,-1] = len(a[0])
    return out


def pos2zyx(ind,sz):
    z = ind//(sz[1]*sz[2])
    y = (ind- z*(sz[1]*sz[2])) // sz[2]
    x = ind % sz[2]
    return [z,y,x]

def zyx2pos(z,y,x,sz):
    return z*sz[1]*sz[2]+y*sz[2]+x
