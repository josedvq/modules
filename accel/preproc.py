import scipy.interpolate
import numpy as np

def get_interp_chunks(time, accel, interp_increment):
    time_chunks = []
    chunks = []
    
    i_from = 0
    for i in range(0, len(time)-1):
        if np.isclose(time[i] + interp_increment, time[i+1]):
            continue
        elif time[i] + interp_increment < time[i+1]:
            # include until i
            time_chunks.append(time[i_from:i+1])
            chunks.append(accel[i_from:i+1,:])
            
            # include interp segment
            t = np.arange(time[i]+interp_increment, time[i+1], interp_increment)
            time_chunks.append(t)
            f = scipy.interpolate.interp1d([time[i], time[i+1]], [accel[i,:],accel[i+1,:]], axis=0)
            chunks.append(f(t))
            
            # update i_from
            i_from = i+1
        else:
            print('time going backwards! at {:f}->{:f}'.format(time[i], time[i+1]))
            
    # add last chunk
    time_chunks.append(time[i_from:i+1])
    chunks.append(accel[i_from:i+1,:])
    
    return time_chunks, chunks#np.concatenate(time_chunks), np.vstack(chunks)

def get_interp_chunks_new(time, accel, interp_increment):
    assert len(time) == len(accel)
    diffs = np.diff(time)
    idxs = np.where(~np.isclose(diffs, 0.05))[0]

    if len(idxs) == 0:
        return [time], [accel]

    time_chunks = []
    chunks = []

    for i in range(len(idxs)):
        # include chunk before
        if i == 0:
            prev_idx = 0
        else:
            prev_idx = idxs[i-1]+1
        time_chunks.append(time[prev_idx: idxs[i]])
        chunks.append(accel[prev_idx: idxs[i], :])

        # include interp chunk
        t = np.arange(time[idxs[i]], time[idxs[i]+1], interp_increment)
        # print(('t', time[idxs[i]], time[idxs[i]+1], len(t)))
        time_chunks.append(t)
        f = scipy.interpolate.interp1d(
            [time[idxs[i]], time[idxs[i]+1]], [accel[idxs[i], :], accel[idxs[i]+1, :]], axis=0)
        chunks.append(f(t))

    prev_idx = idxs[-1]+1
    time_chunks.append(time[prev_idx: len(time)])
    chunks.append(accel[prev_idx: len(time), :])

    assert len(np.concatenate(time_chunks)) == len(time) + int(round(np.sum(diffs[idxs]-0.05) / 0.05))

    return time_chunks, chunks

def interpolate(sig, interp_increment = 0.05):
    '''
    Interpolates missing chunks in a signal sig, where the first column corresponds to time
    '''
    time_chunks, chunks = get_interp_chunks_new(
        sig[:, 0], sig[:, 1:], interp_increment)
    return np.hstack([np.concatenate(time_chunks)[:, None], np.vstack(chunks)])
