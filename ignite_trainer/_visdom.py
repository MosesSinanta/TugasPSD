import os
import sys
import json
import time
import tqdm
import socket
import subprocess
import numpy as np

import visdom

from typing import Tuple
from typing import Optional


def calc_ytick_range(vis: visdom.Visdom, window_name: str, env: Optional[str] = None) -> Tuple[float, float]:
    lower_bound, upper_bound = -1.0, 1.0

    stats = vis.get_window_data(win=window_name, env=env)

    if stats:
        stats = json.loads(stats)

        stats_sanitized = [np.array(item['y']) for item in stats['content']['data'] if 'y' in item]
        stats_sanitized = [item[item != np.array([None])] for item in stats_sanitized if len(item) > 0]
        stats_sanitized = [item.astype(np.float16) for item in stats_sanitized]

        if stats_sanitized:
            # Find the maximum number of columns in the arrays
            max_cols = max(item.shape[0] for item in stats_sanitized)

            # Pad arrays with fewer columns
            stats_sanitized_padded = [np.pad(item, (0, max_cols - item.shape[0]), mode='constant') for item in stats_sanitized]

            stats_sanitized = np.vstack(stats_sanitized_padded)

            q25s = np.quantile(stats_sanitized, 0.25, axis=0)
            q75s = np.quantile(stats_sanitized, 0.75, axis=0)

            iqrs = q75s - q25s

            lower_bounds = q25s - 1.5 * iqrs
            upper_bounds = q75s + 1.5 * iqrs

            q25_sanitized = np.quantile(stats_sanitized, 0.25, axis=0)
            q75_sanitized = np.quantile(stats_sanitized, 0.75, axis=0)

            iqr_sanitized = np.sum(q75_sanitized - q25_sanitized)
            lower_bound = np.min(q25_sanitized) - 1.5 * iqr_sanitized
            upper_bound = np.max(q75_sanitized) + 1.5 * iqr_sanitized

    return lower_bound, upper_bound




def plot_line(vis: visdom.Visdom,
              window_name: str,
              env: Optional[str] = None,
              line_label: Optional[str] = None,
              x: Optional[np.ndarray] = None,
              y: Optional[np.ndarray] = None,
              x_label: Optional[str] = None,
              y_label: Optional[str] = None,
              width: int = 576,
              height: int = 416,
              draw_marker: bool = False) -> str:

    empty_call = not vis.win_exists(window_name)

    if empty_call and (x is not None or y is not None):
        return window_name

    if x is None:
        x = np.ones(1)
        empty_call = empty_call & True

    if y is None:
        y = np.full(1, np.nan)
        empty_call = empty_call & True

        if x.shape != y.shape:
            x = np.ones_like(y)

    opts = {
        'showlegend': True,
        'markers': draw_marker,
        'markersize': 5,
    }

    if empty_call:
        opts['title'] = window_name
        opts['width'] = width
        opts['height'] = height

    window_name = vis.line(
        X=x,
        Y=y,
        win=window_name,
        env=env,
        update='append',
        name=line_label,
        opts=opts
    )

    xtickmin, xtickmax = 0.0, np.max(x) * 1.05
    ytickmin, ytickmax = calc_ytick_range(vis, window_name, env)

    opts = {
        'showlegend': True,
        'xtickmin': xtickmin,
        'xtickmax': xtickmax,
        'ytickmin': ytickmin,
        'ytickmax': ytickmax,
        'xlabel': x_label,
        'ylabel': y_label
    }

    window_name = vis.update_window_opts(win=window_name, opts=opts, env=env)

    return window_name


def create_summary_window(vis: visdom.Visdom,
                          visdom_env_name: str,
                          experiment_name: str,
                          summary: str) -> str:

    return vis.text(
        text=summary,
        win=experiment_name,
        env=visdom_env_name,
        opts={'title': 'Summary', 'width': 576, 'height': 416},
        append=vis.win_exists(experiment_name, visdom_env_name)
    )


def connection_is_alive(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.connect((host, port))
            sock.shutdown(socket.SHUT_RDWR)

            return True
        except socket.error:
            return False


def get_visdom_instance(host: str = 'localhost',
                        port: int = 8097,
                        env_name: str = 'main',
                        env_path: str = 'visdom_env') -> Tuple[visdom.Visdom, Optional[int]]:

    vis_pid = None

    if not connection_is_alive(host, port):
        if any(host.strip('/').endswith(lh) for lh in ['127.0.0.1', 'localhost']):
            os.makedirs(env_path, exist_ok=True)

            tqdm.tqdm.write('Starting visdom on port {}'.format(port), end='')

            vis_args = [
                sys.executable,
                '-m', 'visdom.server',
                '-port', str(port),
                '-env_path', os.path.join(os.getcwd(), env_path)
            ]
            vis_proc = subprocess.Popen(vis_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            time.sleep(2.0)

            vis_pid = vis_proc.pid
            tqdm.tqdm.write('PID -> {}'.format(vis_pid))

    trials_left = 5
    while not connection_is_alive(host, port):
        time.sleep(1.0)

        tqdm.tqdm.write('Trying to connect ({} left)...'.format(trials_left))

        trials_left -= 1
        if trials_left < 1:
            raise RuntimeError('Visdom server is not running. Please run "python -m visdom.server".')

    vis = visdom.Visdom(
        server='http://{}'.format(host),
        port=port,
        env=env_name
    )

    return vis, vis_pid
