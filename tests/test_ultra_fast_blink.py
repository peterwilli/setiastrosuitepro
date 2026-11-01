"""Raw FITS screening, bounded work, review state, and file-action regressions."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import threading
import time

import numpy as np
import pytest
from astropy.io import fits

from setiastro.saspro.ultra_fast_blink_measure import (
    Measurement, RawFitsPlane, make_preview, measure_file, sample_windows,
)


def star_field(shape=(1024, 1024), stars=True, seed=2):
    rng = np.random.default_rng(seed)
    z = rng.normal(1000, 5, shape).astype(np.float32)
    if stars:
        yy, xx = np.mgrid[-7:8, -7:8]
        kernel = 400*np.exp(-(xx**2+yy**2)/(2*1.4**2))
        for y in range(32, shape[0]-16, 64):
            for x in range(32, shape[1]-16, 64):
                z[y-7:y+8, x-7:x+8] += kernel
    return z


def write_frame(tmp_path, name='frame.fits', data=None, header=None):
    path = tmp_path / name
    fits.writeto(path, star_field() if data is None else data, header=header)
    return str(path)


def test_stars_noise_hot_pixels_and_obstruction(tmp_path):
    good = star_field()
    noise = star_field(stars=False)
    hot = noise.copy()
    hot[::31, ::31] += 10000
    partial = good.copy()
    partial[:, :768] = noise[:, :768] + np.linspace(0, 2000, 768)[None, :]
    measurements = [measure_file(write_frame(tmp_path, f'{i}.fits', z))
                    for i, z in enumerate([good, noise, hot, partial])]
    g, n, h, p = measurements
    assert all(m.status == 'ok' for m in measurements)
    assert g.score >= 90 and not g.rejection_reason()
    assert n.score == 0 and n.rejection_reason() == 'No stars detected'
    assert h.score < 10
    assert 0 < p.coverage < g.coverage
    assert p.score < g.score
    assert 2 < g.fwhm < 5


@pytest.mark.parametrize('dtype', [np.int16, np.uint16, np.int32, np.float32, np.float64])
def test_fits_pixel_types(tmp_path, dtype):
    path = write_frame(tmp_path, data=star_field().astype(dtype))
    m = measure_file(path)
    assert m.status == 'ok' and m.score > 90
    image = make_preview(path)
    assert image.dtype == np.uint8 and max(image.shape) <= 1600
    # All file handles/mappings must be gone, including on Windows.
    Path(path).rename(tmp_path/'renamed.fits')


def test_scaling_only_sampled_sections(tmp_path):
    z = star_field().astype(np.int16)
    h = fits.Header({'BSCALE': 2.5, 'BZERO': 8000})
    path = write_frame(tmp_path, data=z, header=h)
    # FITS writer can reset scaling for ndarray input; set it after writing.
    with fits.open(path, mode='update', do_not_scale_image_data=True) as hdul:
        hdul[0].header['BSCALE'] = 2.5
        hdul[0].header['BZERO'] = 8000
    with RawFitsPlane(path) as plane:
        assert plane.data.dtype.kind == 'i'
        patch = plane.patch(0, 0, 16, 16)
        np.testing.assert_allclose(patch, z[:16, :16]*2.5+8000)
        assert patch.flags.owndata
    m = measure_file(path)
    assert m.status == 'ok' and m.score > 90
    assert m.background == pytest.approx(10500, abs=10)


def test_extension_image_and_metadata(tmp_path):
    path = tmp_path/'extension.fit'
    hdr = fits.Header({'EXPTIME': 10, 'GAIN': 100, 'FILTER': 'L'})
    fits.HDUList([fits.PrimaryHDU(), fits.ImageHDU(star_field(), header=hdr)]).writeto(path)
    m = measure_file(path)
    assert m.status == 'ok' and m.score > 90
    assert (m.exposure, m.gain, m.filter_name) == ('10', '100', 'L')


def test_bayer_without_debayer(tmp_path):
    z = star_field()
    for y, x, offset in [(0, 0, 100), (0, 1, 200), (1, 0, 200), (1, 1, 400)]:
        z[y::2, x::2] += offset
    path = write_frame(tmp_path, data=z, header=fits.Header({'BAYERPAT': 'RGGB'}))
    m = measure_file(path)
    assert m.status == 'ok' and m.score > 90
    assert 2 < m.fwhm < 6
    assert make_preview(path).shape == (512, 512)


@pytest.mark.parametrize('shape', [(8, 8), (64, 300), (300, 600), (2180, 3856)])
def test_nonoverlapping_bounded_windows(shape):
    windows = sample_windows(shape)
    assert len(windows) <= 25
    for i, (y, x, h, w) in enumerate(windows):
        assert h <= 256 and w <= 256 and y+h <= shape[0] and x+w <= shape[1]
        for yy, xx, hh, ww in windows[i+1:]:
            assert y+h <= yy or yy+hh <= y or x+w <= xx or xx+ww <= x


def test_small_and_constant(tmp_path):
    m = measure_file(write_frame(tmp_path, data=np.ones((8, 8), np.float32)))
    assert m.status == 'ok' and m.score == 0


@pytest.mark.parametrize('kind', ['cube', 'compressed', 'truncated', 'corrupt', 'unsupported', 'nonfinite', 'blank'])
def test_invalid_inputs_are_errors(tmp_path, kind):
    path = tmp_path/'input.fits'
    if kind == 'cube':
        fits.writeto(path, np.zeros((3, 16, 16), np.float32))
    elif kind == 'compressed':
        fits.HDUList([fits.PrimaryHDU(), fits.CompImageHDU(np.ones((64, 64), np.int16))]).writeto(path)
    elif kind == 'truncated':
        fits.writeto(path, np.ones((128, 128), np.int16))
        with path.open('r+b') as stream:
            stream.truncate(4000)
    elif kind == 'corrupt':
        path.write_bytes(b'not fits')
    elif kind == 'unsupported':
        path = tmp_path/'input.tif'
        path.write_bytes(b'not fits')
    elif kind == 'nonfinite':
        fits.writeto(path, np.full((128, 128), np.nan))
    else:
        fits.writeto(path, np.full((128, 128), -999, np.int16), fits.Header({'BLANK': -999}))
    m = measure_file(path)
    assert m.status == 'error' and m.error
    from setiastro.saspro.ultra_fast_blink import Frame
    assert not Frame(str(path), m).keeper(10)


def test_some_nonfinite_pixels(tmp_path):
    z = star_field()
    z[0, 0] = np.nan
    z[1, 1] = np.inf
    m = measure_file(write_frame(tmp_path, data=z))
    assert m.status == 'ok' and m.score > 90


@pytest.fixture
def qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    return app


def pump(qapp, condition, timeout=5):
    deadline = time.monotonic()+timeout
    while not condition():
        qapp.processEvents()
        assert time.monotonic() < deadline, 'GUI worker did not finish'
        time.sleep(.002)
    qapp.processEvents()


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    from PyQt6.QtCore import QSettings
    from setiastro.saspro import ultra_fast_blink as gui
    monkeypatch.setattr(gui, 'QSettings', lambda: QSettings(str(tmp_path/'prefs.ini'), QSettings.Format.IniFormat))
    w = gui.UltraFastBlinkComparator()
    yield w
    w.close()
    pump(qapp, lambda: not w.scan_thread and not w.preview_thread)
    w.close()


def test_override_threshold_errors_and_stacking(window, tmp_path):
    good = write_frame(tmp_path)
    bad = write_frame(tmp_path, 'bad.fits', star_field(stars=False))
    error = str(tmp_path/'missing.fits')
    window.add_paths([good, bad, error, good])
    assert len(window.frames) == 3
    for path in window.frames:
        window._measured(measure_file(path))
    window.frames[good].measurement = replace(window.frames[good].measurement, score=80)
    assert window.frames[bad].flagged(10)
    assert not window.frames[good].flagged(10)
    window.items[bad].setSelected(True)
    window.toggle_flags()
    window.cutoff.setValue(99)
    assert not window.frames[bad].flagged(99)
    assert window.frames[good].flagged(99)
    window.reset_overrides()
    assert window.frames[bad].flagged(99)
    window.cutoff.setValue(10)
    window.select_keepers()
    assert window.selected_paths() == [good]
    sent = []
    window.sendToStacking.connect(lambda paths, target: sent.append((paths, target)))
    window.send_stacking('lights')
    assert sent[-1] == ([good], 'lights')
    window.tree.clearSelection()
    window.send_stacking('integration')
    assert sent[-1] == (list(window.frames), 'integration')
    window.filter.setCurrentIndex(3)
    assert not window.items[error].isHidden()
    assert window.items[good].isHidden()
    window.clear_flags()
    assert not window.frames[error].keeper(10)


def test_scan_cancel_bounded_jobs_and_close(qapp, window, tmp_path, monkeypatch):
    from setiastro.saspro import ultra_fast_blink as gui
    gate = threading.Event()
    started = []
    def slow(path):
        started.append(path)
        assert gate.wait(3)
        return Measurement(path, score=50)
    monkeypatch.setattr(gui, 'measure_file', slow)
    window.add_paths([str(tmp_path/f'{i}.fits') for i in range(20)])
    window.scan()
    try:
        pump(qapp, lambda: len(started) == 2)
        assert not window.actions_button.isEnabled()
        window.close()
        assert window.closing and window.scan_thread.cancel.is_set()
    finally:
        gate.set()
    pump(qapp, lambda: window.scan_thread is None)
    assert len(started) == 2 and window.completed == 2
    assert sum(f.measurement is None for f in window.frames.values()) == 18


def test_latest_preview_wins_and_cache_is_bounded(qapp, window, tmp_path, monkeypatch):
    from setiastro.saspro import ultra_fast_blink as gui
    started, release = threading.Event(), threading.Event()
    def preview(path):
        if path.endswith('0.fits'):
            started.set()
            assert release.wait(3)
        return np.full((32, 32), int(Path(path).stem), dtype=np.uint8)
    monkeypatch.setattr(gui, 'make_preview', preview)
    paths = [str(tmp_path/f'{i}.fits') for i in range(5)]
    window.add_paths(paths)
    window.tree.setCurrentItem(window.items[paths[0]])
    try:
        pump(qapp, started.is_set)
        window.tree.setCurrentItem(window.items[paths[1]])
        window.tree.setCurrentItem(window.items[paths[2]])
    finally:
        release.set()
    pump(qapp, lambda: window.preview_thread is None)
    assert window.preview_target == paths[2]
    assert paths[1] not in window.preview_cache
    for path in paths[3:]:
        window.tree.setCurrentItem(window.items[path])
        pump(qapp, lambda: window.preview_thread is None)
    assert len(window.preview_cache) == 3
    assert window.preview_pixmap is not None


def test_rename_collision_and_failed_delete_preserve_rows(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QMessageBox
    from setiastro.saspro import ultra_fast_blink as gui
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a: None)
    path = write_frame(tmp_path)
    other = write_frame(tmp_path, 'occupied.fits', np.zeros((8, 8), np.int16))
    window.add_paths([path])
    window._measured(measure_file(path))
    window.frames[path].override = True
    window._rename_pairs([(path, Path(other).name)])
    assert path in window.frames and Path(path).exists()
    window._rename_pairs([(path, '../escape.fits')])
    assert path in window.frames
    window._rename_pairs([(path, 'renamed.fits')])
    renamed = str(tmp_path/'renamed.fits')
    assert renamed in window.frames and path not in window.frames
    assert window.frames[renamed].measurement.path == renamed
    assert window.frames[renamed].override is True
    monkeypatch.setattr(QMessageBox, 'question', lambda *a: QMessageBox.StandardButton.Yes)
    def fail_remove(path):
        raise PermissionError('in use')
    monkeypatch.setattr(gui.os, 'remove', fail_remove)
    window.delete_files(True)
    assert renamed in window.frames and Path(renamed).exists()


def test_move_collision_then_success(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox
    source = tmp_path/'source'
    destination = tmp_path/'destination'
    source.mkdir(); destination.mkdir()
    path = write_frame(source)
    target = Path(write_frame(destination, data=np.zeros((8, 8), np.int16)))
    window.add_paths([path])
    window.frames[path].override = True
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *a: str(destination))
    monkeypatch.setattr(QMessageBox, 'warning', lambda *a: None)
    window.move_files(True)
    assert path in window.frames and Path(path).exists()
    assert fits.getdata(target).shape == (8, 8)
    target.unlink()
    window.move_files(True)
    assert not window.frames and target.exists() and not Path(path).exists()


def test_primary_metadata_with_extension_pixels(tmp_path):
    path = tmp_path/'extension.fts'
    header = fits.Header({'EXPTIME': 60, 'GAIN': 50, 'FILTER': 'Ha'})
    fits.HDUList([fits.PrimaryHDU(header=header), fits.ImageHDU(star_field())]).writeto(path)
    m = measure_file(path)
    assert m.status == 'ok'
    assert (m.exposure, m.gain, m.filter_name) == ('60', '50', 'Ha')


def test_gzip_disguised_as_uncompressed_fits(tmp_path):
    import gzip
    path = Path(write_frame(tmp_path))
    compressed = gzip.compress(path.read_bytes())
    path.write_bytes(compressed)
    m = measure_file(path)
    assert m.status == 'error' and 'Compressed FITS' in m.error


def test_no_clobber_cross_filesystem_and_rollback(tmp_path, monkeypatch):
    import errno
    from setiastro.saspro import ultra_fast_blink as gui
    src, dst = tmp_path/'source', tmp_path/'target'
    src.write_bytes(b'original')
    def cross_device(*args):
        raise OSError(errno.EXDEV, 'different filesystem')
    monkeypatch.setattr(gui.os, 'link', cross_device)
    dst.write_bytes(b'existing')
    with pytest.raises(FileExistsError):
        gui.move_without_overwrite(src, dst)
    assert dst.read_bytes() == b'existing' and src.read_bytes() == b'original'
    dst.unlink()
    real_unlink = gui.os.unlink
    def fail_source_unlink(path, *args, **kwargs):
        if Path(path) == src:
            raise PermissionError('source locked')
        return real_unlink(path, *args, **kwargs)
    monkeypatch.setattr(gui.os, 'unlink', fail_source_unlink)
    with pytest.raises(PermissionError):
        gui.move_without_overwrite(src, dst)
    assert src.read_bytes() == b'original' and not dst.exists()
    monkeypatch.setattr(gui.os, 'unlink', real_unlink)
    gui.move_without_overwrite(src, dst)
    assert not src.exists() and dst.read_bytes() == b'original'


def test_scan_then_rescan_keeps_override(qapp, window, tmp_path):
    path = write_frame(tmp_path, data=star_field(stars=False))
    window.add_paths([path])
    window.scan()
    pump(qapp, lambda: window.scan_thread is None)
    assert window.frames[path].flagged(10)
    window.frames[path].override = False
    window.scan()
    pump(qapp, lambda: window.scan_thread is None)
    assert window.frames[path].keeper(10)


def test_numeric_sort_and_hidden_selection(window, tmp_path):
    paths = [str(tmp_path/f'{i}.fits') for i in range(3)]
    window.add_paths(paths)
    for p, score in zip(paths, [4, 80, 12]):
        window._measured(Measurement(p, score=score))
    from PyQt6.QtCore import Qt
    window.tree.sortItems(1, Qt.SortOrder.AscendingOrder)
    assert [float(window.tree.topLevelItem(i).text(1)) for i in range(3)] == [4, 12, 80]
    window.items[paths[0]].setSelected(True)
    window.filter.setCurrentIndex(1)
    assert not window.selected_paths()


def test_folder_discovery_and_permission_error(window, tmp_path, monkeypatch):
    from PyQt6.QtWidgets import QFileDialog, QMessageBox
    folder = tmp_path/'input'
    folder.mkdir()
    nested = folder/'nested'
    nested.mkdir()
    a = write_frame(folder, 'a.FITS', np.ones((8, 8), np.int16))
    b = write_frame(nested, 'b.fit', np.ones((8, 8), np.int16))
    compressed = folder/'compressed.fits.gz'
    compressed.write_bytes(b'unsupported')
    (folder/'notes.txt').write_text('not an image')
    monkeypatch.setattr(QFileDialog, 'getExistingDirectory', lambda *a: str(folder))
    window.add_folder()
    assert set(window.frames) == {a, str(compressed)}
    window.recursive.setChecked(True)
    window.add_folder()
    assert set(window.frames) == {a, b, str(compressed)}
    failures = []
    monkeypatch.setattr(QMessageBox, 'warning', lambda *args: failures.append(args[-1]))
    def fail_listing(*args):
        raise PermissionError('Cannot list directory')
    window.recursive.setChecked(False)
    monkeypatch.setattr(Path, 'iterdir', fail_listing)
    window.add_folder()
    assert failures == ['Cannot list directory']
