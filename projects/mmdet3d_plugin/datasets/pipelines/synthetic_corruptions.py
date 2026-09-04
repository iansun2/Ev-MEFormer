import math

import numpy as np
import torch
from mmdet.datasets.builder import PIPELINES

try:
    import cv2
except ImportError:  # pragma: no cover - training env usually has cv2 via mmcv.
    cv2 = None


@PIPELINES.register_module()
class OnlineSyntheticMultiCorrupt(object):
    """Lightweight online sensor corruptions for train-only robustness probes.

    The operators follow the official MultiCorrupt converter severity mapping
    where it can be applied online. Expensive offline-only effects such as
    LiDAR fog lookup tables, LiDAR snow particles, and previous-frame temporal
    replacement are approximated without extra dataset files.
    """

    _VALID_CORRUPTIONS = (
        'pointsreducing',
        'beamsreducing',
        'spatialmisalignment',
        'temporalmisalignment',
        'brightness',
        'dark',
        'fog',
        'motionblur',
        'missingcamera',
        'snow',
    )

    def __init__(self,
                 corrupt_prob=0.30,
                 severity_choices=(1, 2, 3),
                 corruption_weights=None,
                 num_beams=32,
                 record_meta=True):
        self.corrupt_prob = float(corrupt_prob)
        self.severity_choices = tuple(int(s) for s in severity_choices)
        self.num_beams = int(num_beams)
        self.record_meta = bool(record_meta)

        if not 0.0 <= self.corrupt_prob <= 1.0:
            raise ValueError('corrupt_prob must be in [0, 1]')
        if not self.severity_choices:
            raise ValueError('severity_choices must be non-empty')
        if self.num_beams <= 0:
            raise ValueError('num_beams must be positive')

        if corruption_weights is None:
            corruption_weights = dict(
                pointsreducing=0.25,
                beamsreducing=0.25,
                spatialmisalignment=0.20,
                temporalmisalignment=0.10,
                dark=0.10,
                motionblur=0.07,
                missingcamera=0.03,
            )

        names = []
        weights = []
        for name, weight in corruption_weights.items():
            if name not in self._VALID_CORRUPTIONS:
                raise ValueError(f'Unsupported corruption: {name}')
            if weight < 0:
                raise ValueError(f'Negative weight for corruption: {name}')
            if weight > 0:
                names.append(name)
                weights.append(float(weight))
        if not names:
            raise ValueError('At least one corruption weight must be positive')
        weights = np.asarray(weights, dtype=np.float64)
        self.corruption_names = tuple(names)
        self.corruption_probs = weights / weights.sum()

    def __call__(self, results):
        corruption_type = 'clean'
        severity = 0
        applied = False

        if np.random.rand() < self.corrupt_prob:
            corruption_type = np.random.choice(
                self.corruption_names, p=self.corruption_probs)
            severity = int(np.random.choice(self.severity_choices))
            getattr(self, f'_apply_{corruption_type}')(results, severity)
            applied = True

        if self.record_meta:
            results['synthetic_corruption'] = applied
            results['corruption_type'] = corruption_type
            results['corruption_severity'] = severity
        return results

    def _set_points(self, results, mask):
        points = results.get('points')
        if points is None:
            return
        if mask.numel() == 0:
            return
        if mask.sum().item() == 0:
            keep_idx = torch.randint(mask.numel(), (1,), device=mask.device)
            mask[keep_idx] = True
        results['points'] = points[mask]

    def _points_tensor(self, results):
        points = results.get('points')
        if points is None:
            return None
        return points.tensor

    def _keep_mask(self, num_points, keep_ratio, device):
        mask = torch.rand(num_points, device=device) < keep_ratio
        if num_points > 0 and mask.sum().item() == 0:
            mask[torch.randint(num_points, (1,), device=device)] = True
        return mask

    def _looks_like_beam_channel(self, values):
        if values.numel() == 0:
            return False
        finite = torch.isfinite(values)
        if finite.sum().item() == 0:
            return False
        values = values[finite]
        if values.min().item() < 0 or values.max().item() > 63:
            return False
        rounded = torch.round(values)
        integer_like = torch.mean((torch.abs(values - rounded) < 1e-3).float())
        unique_count = torch.unique(rounded.long()).numel()
        return integer_like.item() > 0.98 and unique_count > 3

    def _pseudo_beam_ids_from_elevation(self, tensor):
        xy_norm = torch.norm(tensor[:, :2], dim=1).clamp_min(1e-6)
        elev = torch.atan2(tensor[:, 2], xy_norm)
        min_elev = elev.min()
        max_elev = elev.max()
        span = (max_elev - min_elev).clamp_min(1e-6)
        beam_ids = torch.floor(
            (elev - min_elev) / span * self.num_beams).long()
        return beam_ids.clamp_(0, self.num_beams - 1)

    def _apply_lidar_motion(self, results, severity):
        tensor = self._points_tensor(results)
        if tensor is None or tensor.shape[0] == 0:
            return
        std = {1: 0.06, 2: 0.10, 3: 0.13}.get(severity, 0.13)
        translate = tensor.new_tensor(np.random.normal(0.0, std, size=3))
        num_points = tensor.shape[0]
        jitter_x = torch.clamp(
            torch.randn(num_points, device=tensor.device, dtype=tensor.dtype)
            * (std * 0.15), -3 * std, 3 * std)
        jitter_y = torch.clamp(
            torch.randn(num_points, device=tensor.device, dtype=tensor.dtype)
            * (std * 0.20), -3 * std, 3 * std)
        jitter_z = torch.clamp(
            torch.randn(num_points, device=tensor.device, dtype=tensor.dtype)
            * (std * 0.12), -3 * std, 3 * std)
        tensor[:, :3] += translate
        tensor[:, 0] += jitter_x
        tensor[:, 1] += jitter_y
        tensor[:, 2] += jitter_z

    def _apply_pointsreducing(self, results, severity):
        keep_ratio = {1: 0.30, 2: 0.20, 3: 0.10}.get(severity, 0.10)
        tensor = self._points_tensor(results)
        if tensor is None:
            return
        mask = self._keep_mask(tensor.shape[0], keep_ratio, tensor.device)
        self._set_points(results, mask)

    def _apply_beamsreducing(self, results, severity):
        allowed = {
            1: (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31),
            2: (1, 5, 9, 13, 17, 21, 25, 29),
            3: (1, 9, 17, 25),
        }.get(severity, (1, 9, 17, 25))
        tensor = self._points_tensor(results)
        if tensor is None:
            return
        if tensor.shape[0] == 0:
            return

        if tensor.shape[1] > 4 and self._looks_like_beam_channel(tensor[:, 4]):
            beam_ids = torch.round(tensor[:, 4]).long()
        else:
            beam_ids = self._pseudo_beam_ids_from_elevation(tensor)

        allowed_tensor = torch.tensor(allowed, device=tensor.device)
        mask = (beam_ids[:, None] == allowed_tensor[None, :]).any(dim=1)
        if mask.sum().item() == 0:
            keep_ratio = len(allowed) / float(max(self.num_beams, 1))
            mask = self._keep_mask(tensor.shape[0], keep_ratio, tensor.device)
        self._set_points(results, mask)

    def _apply_temporalmisalignment(self, results, severity):
        prob = {1: 0.20, 2: 0.40, 3: 0.60}.get(severity, 0.60)
        if np.random.rand() >= prob:
            return

        tensor = self._points_tensor(results)
        if tensor is not None and tensor.shape[0] > 0:
            if tensor.shape[1] >= 5:
                non_keyframe = torch.abs(tensor[:, 4]) > 1e-3
                keep_non_key = {1: 0.80, 2: 0.60, 3: 0.40}.get(severity, 0.40)
                rand = torch.rand(tensor.shape[0], device=tensor.device)
                mask = ~non_keyframe | (rand < keep_non_key)
                self._set_points(results, mask)
            self._apply_lidar_motion(results, severity)

        max_shift = {1: 4, 2: 8, 3: 12}.get(severity, 12)
        new_imgs = []
        for img in results.get('img', []):
            dx = int(np.random.randint(-max_shift, max_shift + 1))
            dy = int(np.random.randint(-max_shift, max_shift + 1))
            new_imgs.append(self._shift_image(img, dx, dy))
        if new_imgs:
            results['img'] = new_imgs

    def _apply_spatialmisalignment(self, results, severity):
        prob, angle_deg = {
            1: (0.20, 1.0),
            2: (0.40, 2.0),
            3: (0.60, 3.0),
        }.get(severity, (0.60, 3.0))
        if np.random.rand() >= prob:
            return

        tensor = self._points_tensor(results)
        if tensor is None or tensor.shape[0] == 0:
            return

        theta = math.radians(angle_deg)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)
        rot = tensor.new_tensor([
            [1.0, 0.0, 0.0],
            [0.0, cos_t, sin_t],
            [0.0, -sin_t, cos_t],
        ]) @ tensor.new_tensor([
            [cos_t, 0.0, -sin_t],
            [0.0, 1.0, 0.0],
            [sin_t, 0.0, cos_t],
        ]) @ tensor.new_tensor([
            [cos_t, sin_t, 0.0],
            [-sin_t, cos_t, 0.0],
            [0.0, 0.0, 1.0],
        ])
        tensor[:, :3] = tensor[:, :3] @ rot.t()
        tensor[:, 0] += 2.0

    def _apply_dark(self, results, severity):
        max_value = {1: 0.60, 2: 0.40, 3: 0.30}.get(severity, 0.30)
        new_imgs = []
        for img in results.get('img', []):
            dtype = img.dtype
            out = np.asarray(img, dtype=np.float32) / 255.0
            lo = float(out.min())
            hi = float(out.max())
            if hi > lo:
                out = np.power((out - lo) / (hi - lo), 2.0) * max_value
            else:
                out = np.zeros_like(out)
            out = self._poisson_gaussian_noise(out * 255.0, severity)
            new_imgs.append(out.astype(dtype, copy=False))
        results['img'] = new_imgs

    def _apply_brightness(self, results, severity):
        delta = {1: 0.50, 2: 0.60, 3: 0.70}.get(severity, 0.70)
        new_imgs = []
        for img in results.get('img', []):
            dtype = img.dtype
            out = np.asarray(img, dtype=np.float32)
            if cv2 is not None and out.ndim == 3 and out.shape[2] >= 3:
                hsv = cv2.cvtColor(
                    np.clip(out, 0, 255).astype(np.uint8),
                    cv2.COLOR_BGR2HSV).astype(np.float32)
                hsv[:, :, 2] = np.clip(hsv[:, :, 2] / 255.0 + delta, 0, 1) * 255.0
                out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
            else:
                out = np.clip(out + delta * 255.0, 0.0, 255.0)
            new_imgs.append(out.astype(dtype, copy=False))
        results['img'] = new_imgs

    def _apply_fog(self, results, severity):
        self._apply_lidar_fog(results, severity)
        new_imgs = []
        for img in results.get('img', []):
            dtype = img.dtype
            out = self._image_fog(img, severity)
            new_imgs.append(np.clip(out, 0.0, 255.0).astype(dtype, copy=False))
        results['img'] = new_imgs

    def _apply_motionblur(self, results, severity):
        self._apply_lidar_motion(results, severity)
        sigma = {1: 0.06, 2: 0.10, 3: 0.13}.get(severity, 0.13)
        new_imgs = []
        for img in results.get('img', []):
            dtype = img.dtype
            angle = np.random.uniform(-45.0, 45.0)
            out = self._motion_blur_image(img, radius=15, sigma=sigma, angle=angle)
            new_imgs.append(np.asarray(out).astype(dtype, copy=False))
        results['img'] = new_imgs

    def _motion_blur_image(self, img, radius, sigma, angle):
        width = radius * 2 + 1
        x = np.arange(width, dtype=np.float32)
        kernel = np.exp(-(x ** 2) / (2 * (sigma * 100) ** 2))
        kernel /= max(float(kernel.sum()), 1e-6)
        point = (width * math.sin(math.radians(angle)),
                 width * math.cos(math.radians(angle)))
        hypot = max(math.hypot(point[0], point[1]), 1e-6)
        img_float = np.asarray(img, dtype=np.float32)
        blurred = np.zeros_like(img_float, dtype=np.float32)
        for i in range(width):
            dy = -math.ceil(((i * point[0]) / hypot) - 0.5)
            dx = -math.ceil(((i * point[1]) / hypot) - 0.5)
            if abs(dy) >= img_float.shape[0] or abs(dx) >= img_float.shape[1]:
                break
            blurred += kernel[i] * self._shift_image(img_float, dx, dy)
        return np.clip(blurred, 0.0, 255.0)

    def _shift_image(self, img, dx, dy):
        shifted = np.asarray(img).copy()
        if dx < 0:
            shifted = np.roll(shifted, shift=shifted.shape[1] + dx, axis=1)
            shifted[:, dx:] = shifted[:, dx - 1:dx]
        elif dx > 0:
            shifted = np.roll(shifted, shift=dx, axis=1)
            shifted[:, :dx] = shifted[:, dx:dx + 1]
        if dy < 0:
            shifted = np.roll(shifted, shift=shifted.shape[0] + dy, axis=0)
            shifted[dy:, :] = shifted[dy - 1:dy, :]
        elif dy > 0:
            shifted = np.roll(shifted, shift=dy, axis=0)
            shifted[:dy, :] = shifted[dy:dy + 1, :]
        return shifted

    def _poisson_gaussian_noise(self, img, severity):
        poisson_scale = 10 * {1: 25, 2: 12, 3: 5}.get(severity, 5)
        x = np.clip(np.asarray(img, dtype=np.float32) / 255.0, 0.0, 1.0)
        x = np.random.poisson(x * poisson_scale) / float(poisson_scale)
        gauss_scale = 0.1 * {1: 0.08, 2: 0.18, 3: 0.26}.get(severity, 0.26)
        x = np.clip(x + np.random.normal(size=x.shape, scale=gauss_scale), 0.0, 1.0)
        return x * 255.0

    def _apply_lidar_fog(self, results, severity):
        tensor = self._points_tensor(results)
        if tensor is None or tensor.shape[0] == 0 or tensor.shape[1] < 4:
            return
        alpha, beta = {
            1: (0.02, 0.008),
            2: (0.03, 0.008),
            3: (0.06, 0.05),
        }.get(severity, (0.06, 0.05))
        dist = torch.norm(tensor[:, :3], dim=1).clamp_min(1e-3)
        attenuation = torch.exp(-2.0 * alpha * dist).to(tensor.dtype)
        tensor[:, 3] *= attenuation

        # Online approximation of MultiCorrupt's LiDAR fog soft targets:
        # move a small subset of low-return far points closer to the sensor.
        fog_prob = torch.clamp(
            tensor.new_tensor(beta * 12.0) * (dist / 80.0), 0.0, 0.35)
        fog_mask = torch.rand(tensor.shape[0], device=tensor.device) < fog_prob
        if fog_mask.any():
            scale = torch.empty(
                int(fog_mask.sum().item()),
                device=tensor.device,
                dtype=tensor.dtype).uniform_(0.35, 0.85)
            tensor[fog_mask, :3] *= scale[:, None]

    def _image_fog(self, img, severity):
        beta = {1: 0.0005, 2: 0.0010, 3: 0.0020}.get(severity, 0.0020)
        fog_strength = {1: 2.0, 2: 3.0, 3: 4.0}.get(severity, 4.0)
        img_float = np.asarray(img, dtype=np.float32)
        shape = img_float.shape
        max_side = max(shape[0], shape[1])
        map_size = self._next_power_of_2(max_side)

        fog = self._plasma_fractal(map_size, wibbledecay=1.7)[:shape[0], :shape[1]]
        attenuation_map = self._plasma_fractal(map_size)[:shape[0], :shape[1]]
        backscatter_map = self._plasma_fractal(map_size)[:shape[0], :shape[1]]
        fog *= (1.0 - 0.8 * attenuation_map)
        fog *= (1.0 + 0.5 * backscatter_map)

        x = img_float / 255.0
        max_val = max(float(x.max()), 1e-6)
        if img_float.ndim == 3:
            x = x + fog_strength * fog[..., None]
        else:
            x = x + fog_strength * fog
        x = np.clip(x * max_val / (max_val + fog_strength), 0.0, 1.0)

        rows, cols = shape[:2]
        y = np.arange(rows, dtype=np.float32)[:, None]
        xcoord = np.arange(cols, dtype=np.float32)[None, :]
        y1, x1 = int(rows * 0.65), int(cols * 0.1)
        y2, x2 = int(rows * 0.99), int(cols * 0.85)
        d_rect = np.sqrt(
            np.maximum(0, np.maximum(y - y2, y1 - y)) ** 2 +
            np.maximum(0, np.maximum(xcoord - x2, x1 - xcoord)) ** 2)
        d = 2.0 * d_rect * (1.0 + np.random.rand(rows, cols) / 4.0)
        d += math.sqrt(max(rows, cols))
        transmission = np.exp(-beta * d).astype(np.float32)
        if img_float.ndim == 3:
            transmission = transmission[..., None]
        return (x * transmission + 0.5 * (1.0 - transmission)) * 255.0

    def _next_power_of_2(self, value):
        value = int(value)
        return 1 if value <= 1 else 2 ** (value - 1).bit_length()

    def _plasma_fractal(self, mapsize=256, wibbledecay=1.7):
        maparray = np.empty((mapsize, mapsize), dtype=np.float32)
        maparray[0, 0] = 0.0
        stepsize = mapsize
        wibble = 100.0

        while stepsize >= 2:
            half = stepsize // 2
            corner = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            square = corner + np.roll(corner, -1, axis=0)
            square += np.roll(square, -1, axis=1)
            maparray[half:mapsize:stepsize, half:mapsize:stepsize] = (
                square / 4.0 + wibble * np.random.uniform(
                    -wibble, wibble, square.shape))

            center = maparray[half:mapsize:stepsize, half:mapsize:stepsize]
            corner = maparray[0:mapsize:stepsize, 0:mapsize:stepsize]
            ldrsum = center + np.roll(center, 1, axis=0)
            lulsum = corner + np.roll(corner, -1, axis=1)
            maparray[0:mapsize:stepsize, half:mapsize:stepsize] = (
                (ldrsum + lulsum) / 4.0 +
                wibble * np.random.uniform(-wibble, wibble, center.shape))

            tdrsum = center + np.roll(center, 1, axis=1)
            tulsum = corner + np.roll(corner, -1, axis=0)
            maparray[half:mapsize:stepsize, 0:mapsize:stepsize] = (
                (tdrsum + tulsum) / 4.0 +
                wibble * np.random.uniform(-wibble, wibble, center.shape))

            stepsize //= 2
            wibble /= wibbledecay

        maparray -= maparray.min()
        denom = max(float(maparray.max()), 1e-6)
        return maparray / denom

    def _rolling_blur(self, img, kernel_size, axis):
        img_float = np.asarray(img, dtype=np.float32)
        offsets = range(-(kernel_size // 2), kernel_size // 2 + 1)
        acc = np.zeros_like(img_float)
        for offset in offsets:
            acc += np.roll(img_float, shift=offset, axis=axis)
        return np.clip(acc / kernel_size, 0.0, 255.0)

    def _box_blur(self, img, kernel_size):
        img_float = np.asarray(img, dtype=np.float32)
        offsets = range(-(kernel_size // 2), kernel_size // 2 + 1)
        acc = np.zeros_like(img_float)
        count = 0
        for y_offset in offsets:
            for x_offset in offsets:
                acc += np.roll(
                    np.roll(img_float, shift=y_offset, axis=0),
                    shift=x_offset,
                    axis=1)
                count += 1
        return np.clip(acc / max(count, 1), 0.0, 255.0)

    def _apply_snow(self, results, severity):
        self._apply_lidar_snow(results, severity)
        new_imgs = []
        for img in results.get('img', []):
            dtype = img.dtype
            out = self._image_snow(img, severity)
            new_imgs.append(np.clip(out, 0.0, 255.0).astype(dtype, copy=False))
        results['img'] = new_imgs

    def _apply_missingcamera(self, results, severity):
        prob = {1: 0.20, 2: 0.40, 3: 0.60}.get(severity, 0.60)
        imgs = list(results.get('img', []))
        if not imgs:
            return
        for idx, img in enumerate(imgs):
            if np.random.rand() < prob:
                imgs[idx] = np.zeros_like(imgs[idx])
        results['img'] = imgs

    def _apply_lidar_snow(self, results, severity):
        tensor = self._points_tensor(results)
        if tensor is None or tensor.shape[0] == 0:
            return
        keep_ratio = {1: 0.92, 2: 0.84, 3: 0.76}.get(severity, 0.76)
        dist = torch.norm(tensor[:, :3], dim=1)
        far_factor = torch.clamp(dist / 80.0, 0.0, 1.0)
        keep_prob = torch.clamp(
            tensor.new_tensor(keep_ratio) - 0.10 * far_factor, 0.50, 1.0)
        mask = torch.rand(tensor.shape[0], device=tensor.device) < keep_prob
        self._set_points(results, mask)
        tensor = self._points_tensor(results)
        if tensor is not None and tensor.shape[0] > 0 and tensor.shape[1] >= 4:
            attenuation = {1: 0.95, 2: 0.88, 3: 0.80}.get(severity, 0.80)
            tensor[:, 3] *= attenuation

    def _image_snow(self, img, severity):
        params = {
            1: (0.3, 5, 0.90, 7, 7, 0.90),
            2: (0.4, 4, 0.85, 8, 8, 0.85),
            3: (0.5, 3, 0.80, 10, 10, 0.80),
        }.get(severity, (0.5, 3, 0.80, 10, 10, 0.80))
        loc, zoom_factor, threshold, radius, sigma, blend = params
        x = np.asarray(img, dtype=np.float32) / 255.0
        snow_layer = np.random.normal(size=x.shape[:2], loc=loc, scale=0.3)
        snow_layer = self._resize_noise_layer(snow_layer, zoom_factor, x.shape[:2])
        snow_layer[snow_layer < threshold] = 0
        snow_layer = np.clip(snow_layer, 0.0, 1.0)
        snow_layer = self._motion_blur_image(
            snow_layer[..., None] * 255.0,
            radius=radius,
            sigma=sigma,
            angle=np.random.uniform(-135.0, -45.0))[:, :, 0] / 255.0
        snow_layer = np.round(snow_layer * 255.0).astype(np.uint8) / 255.0
        snow_layer = snow_layer[..., None]
        gray = x.mean(axis=2, keepdims=True) if x.ndim == 3 else x[..., None]
        lifted = np.maximum(x, gray * 1.5 + 0.5)
        image = (blend * x + (1.0 - blend) * lifted) * 0.9
        snow = snow_layer + np.rot90(snow_layer, k=2)
        return np.clip(image + snow, 0.0, 1.0) * 255.0

    def _resize_noise_layer(self, layer, zoom_factor, target_shape):
        crop_h = int(np.ceil(layer.shape[0] / float(zoom_factor)))
        crop_w = int(np.ceil(layer.shape[1] / float(zoom_factor)))
        top = (layer.shape[0] - crop_h) // 2
        left = (layer.shape[1] - crop_w) // 2
        cropped = layer[top:top + crop_h, left:left + crop_w]
        if cv2 is not None:
            resized = cv2.resize(
                cropped.astype(np.float32),
                (target_shape[1], target_shape[0]),
                interpolation=cv2.INTER_LINEAR)
        else:
            repeat_h = int(np.ceil(target_shape[0] / max(cropped.shape[0], 1)))
            repeat_w = int(np.ceil(target_shape[1] / max(cropped.shape[1], 1)))
            resized = np.repeat(np.repeat(cropped, repeat_h, axis=0), repeat_w, axis=1)
            resized = resized[:target_shape[0], :target_shape[1]]
        return resized

    def __repr__(self):
        return (
            f'{self.__class__.__name__}('
            f'corrupt_prob={self.corrupt_prob}, '
            f'severity_choices={self.severity_choices}, '
            f'corruptions={self.corruption_names})')
