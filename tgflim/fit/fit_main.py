# fit/fitter.py
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from gen.configs import Config 
from fit import exp, rld        

# registry for fitter methods
# will also use this to tell user the allowable methods
FIT_FUNCS = {
    "mono": exp.mono,
    "bi": exp.bi,
    "mono_rld": rld.mono_rld,
    "bi_rld": rld.bi_rld,
}

def fit_worker(args):
    # pickleable function for ProcessPoolExecutor
    trace, times, fit_name, cfg_dict = args
    fn = FIT_FUNCS.get(fit_name, None)
    if fn is None:
        return None
    return fn(times[:len(trace)], trace, **cfg_dict)

class Fitter:
    def __init__(self, config: Config):
        self.config = config

    def fit_pixel(self, trace):
        if np.sum(trace) < self.config.thresh:
            return None
        
        fn = FIT_FUNCS.get(self.config.fit, None)
        if fn is None:
            return None
        return fn(self.cfg.times[:len(trace)], trace, **asdict(self.cfg))

    def _ensure_maps(self, results, sample_dict, x, y):
        if results:
            return results
        
        # lazy-create maps using keys from the first successful fit
        return {k: np.zeros((x, y), dtype=float) for k in sample_dict.keys()}

    def fit_image(self, data, max_workers: int = 1):
        ''' data: (numsteps, x, y) uint/int/float '''
        numsteps, x, y = data.shape
        results = {}

        if max_workers <= 1:
            # serial
            for i in range(x):
                for j in range(y):
                    trace = data[:, i, j]
                    fitres = self.fit_pixel(trace)
                    if fitres:
                        if not results:
                            results = self._ensure_maps(results, fitres, x, y)
                        for k, v in fitres.items():
                            results[k][i, j] = v
            return results

        # parallel 
        cfg_dict = asdict(self.cfg)
        tasks = []
        for i in range(x):
            for j in range(y):
                trace = data[:, i, j]
                if np.sum(trace) >= self.cfg.thresh:
                    tasks.append((trace, self.cfg.times, self.cfg.fit, cfg_dict))

        with ProcessPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(fit_worker, t) for t in tasks]
            idx = 0
            for i in range(x):
                for j in range(y):
                    trace = data[:, i, j]
                    if np.sum(trace) < self.cfg.thresh:
                        continue
                    fitres = futures[idx].result()
                    idx += 1
                    if fitres:
                        if not results:
                            results = self._ensure_maps(results, fitres, x, y)
                        for k, v in fitres.items():
                            results[k][i, j] = v

        return results
