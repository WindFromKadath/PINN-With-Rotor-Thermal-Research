import numpy as np


def inputs(frame, p, kind):
    r, z = frame.r.to_numpy()/p.L, frame.z.to_numpy()/p.L
    t = frame.t.to_numpy()/p.tau
    angle = frame.theta.to_numpy() if "theta" in frame else np.zeros(len(frame))
    if kind == "thermal":
        return np.column_stack((r, z, t, np.sin(angle), np.cos(angle)))
    if kind == "steady":
        return np.column_stack((r, z, np.sin(angle), np.cos(angle)))
    if kind == "iwg":
        return np.column_stack((r, z))
    return np.column_stack((r, z, t))


def targets(frame, p, kind):
    if kind in ("thermal", "steady"):
        return (frame[["T"]].to_numpy()-p.T0)/p.dT
    return frame[["u_r", "u_z"]].to_numpy()/p.U


def physical(output, p, kind):
    return output*p.dT+p.T0 if kind in ("thermal", "steady") else output*p.U
