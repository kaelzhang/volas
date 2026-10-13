"""Typed comparisons and integer zero survive every continuation owner."""
import numpy as np
import pytest
import volas

@pytest.mark.parametrize("d,kind,last", [
    ('ema:2/(-i)>0', 'int64', 0), ('ema:2/(-i)>0', 'int32', 0),
    ('ema:2/(-i)>0', 'bool', False), ('(ema:2>0)&(b<i)', 'int64', 2**53+1),
    ('(ema:2>0)&(b==i)', 'int64', 2**53+1),
])
@pytest.mark.parametrize("mode", ['append','batch','replace','write','slice','window','tf','tf_window'])
def test_typed_continuation(d, kind, last, mode):
    n=20;data={'close':np.arange(1.,n+1),'i':np.ones(n,dtype=kind),'b':np.ones(n,dtype=bool)}
    tf=mode.startswith('tf');idx=(np.datetime64('2025-01-01','ns')+np.arange(n)*np.timedelta64(5,'m')) if tf else np.arange(n)
    kw={'window':5,'max_lookback':[d]} if 'window' in mode else {}
    if tf:kw['time_frame']='5m'

    f=volas.DataFrame(data,index=idx,**kw);f[d]
    if mode=='slice':f=f.iloc[5:]
    if mode=='write':
     f.iat[len(f)-1,f.columns.index('i')]=last;data['i'][-1]=last
    elif mode=='replace' or tf:
     ts=idx[-1]+np.timedelta64(1,'m') if tf else idx[-1]
     row={'close':[20.],'i':np.array([last],dtype=kind),'b':[True]}
     f.append(volas.DataFrame(row,index=[ts]));data['i'][-1]=last
    else:
     j=2 if mode=='batch' else 1
     row={'close':np.arange(21.,21+j),'i':np.full(j,last,dtype=kind),'b':np.ones(j,dtype=bool)}
     f.append(volas.DataFrame(row,index=np.arange(20,20+j)))
     data={k:np.r_[v,row[k]] for k,v in data.items()}
    got=f[d].to_numpy(dtype='float64')
    want=volas.DataFrame(data).exec(d,dtype='float64')[-len(f):]
    assert ((got.view('uint64')==want.view('uint64'))|(np.isnan(got)&np.isnan(want))).all()
