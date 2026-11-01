"""Ultra Fast Blink Comparator: raw FITS screening with on-demand previews."""
from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, replace
import math
import errno
import os
from pathlib import Path
import shutil
import threading
from time import perf_counter

from PyQt6.QtCore import Qt, QThread, QSettings, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QCheckBox,
    QComboBox, QDoubleSpinBox, QTreeWidget, QTreeWidgetItem, QSplitter,
    QAbstractItemView, QProgressBar, QFileDialog, QMessageBox, QMenu,
    QInputDialog, QDialog, QDialogButtonBox, QLineEdit, QFormLayout,
)

from setiastro.saspro.ultra_fast_blink_measure import (
    DEFAULT_CUTOFF, FITS_SUFFIXES, Measurement, make_preview, measure_file,
)


@dataclass
class Frame:
    path: str
    measurement: Measurement | None = None
    override: bool | None = None  # True = reject, False = keep, None = automatic

    def flagged(self, cutoff):
        if self.override is not None:
            return self.override
        return bool(self.measurement and self.measurement.status == 'ok'
                    and self.measurement.score < cutoff)

    def keeper(self, cutoff):
        return bool(self.measurement and self.measurement.status == 'ok'
                    and not self.flagged(cutoff))


def move_without_overwrite(source, destination):
    """Reserve the destination atomically, including moves across filesystems."""
    created = False
    try:
        try:
            os.link(source, destination)
            created = True
        except OSError as exc:
            if exc.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                raise
            with open(source, 'rb') as src, open(destination, 'xb') as dst:
                created = True
                shutil.copyfileobj(src, dst, length=1024*1024)
            shutil.copystat(source, destination)
        os.unlink(source)
    except Exception:
        if created:
            os.unlink(destination)
        raise


class ScanThread(QThread):
    measured = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, paths, parent=None):
        super().__init__(parent)
        self.paths = paths
        self.cancel = threading.Event()

    def run(self):
        try:
            # At most two jobs in flight, including reads; no unbounded future queue.
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix='ultra-blink') as pool:
                paths = iter(self.paths)
                pending = set()
                for _ in range(2):
                    if (path := next(paths, None)) is not None and not self.cancel.is_set():
                        pending.add(pool.submit(measure_file, path))
                while pending:
                    done, pending = wait(pending, return_when=FIRST_COMPLETED)
                    for future in done:
                        self.measured.emit(future.result())
                        if not self.cancel.is_set() and (path := next(paths, None)) is not None:
                            pending.add(pool.submit(measure_file, path))
        except Exception as exc:
            self.failed.emit(str(exc))


class PreviewThread(QThread):
    ready = pyqtSignal(str, object, str)

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.path = path

    def run(self):
        try:
            self.ready.emit(self.path, make_preview(self.path), '')
        except Exception as exc:
            self.ready.emit(self.path, None, str(exc))


class SortItem(QTreeWidgetItem):
    def __lt__(self, other):
        column = self.treeWidget().sortColumn()
        a, b = self.data(column, Qt.ItemDataRole.UserRole + 1), other.data(column, Qt.ItemDataRole.UserRole + 1)
        if a is not None and b is not None:
            return a < b
        return self.text(column).casefold() < other.text(column).casefold()


class UltraFastBlinkComparator(QWidget):
    sendToStacking = pyqtSignal(list, str)

    def __init__(self, doc_manager=None, parent=None):
        super().__init__(parent)
        self.doc_manager = doc_manager
        self.frames: dict[str, Frame] = {}
        self.items = {}
        self.scan_thread = None
        self.preview_thread = None
        self.preview_target = None
        self.preview_cache = OrderedDict()
        self.preview_pixmap = None
        self.closing = False
        self.completed = 0
        self.started = 0.0
        self.scan_failure = ''
        self.settings = QSettings()
        self.setWindowTitle(self.tr('Ultra Fast Blink Comparator'))
        self.resize(1300, 800)
        self._build_ui()
        QApplication.instance().aboutToQuit.connect(self.shutdown_workers)
        geometry = self.settings.value('ultra_blink/geometry')
        if geometry:
            self.restoreGeometry(geometry)

    def _button(self, layout, text, callback):
        button = QPushButton(self.tr(text))
        button.clicked.connect(callback)
        layout.addWidget(button)
        return button

    def _build_ui(self):
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        layout.addLayout(top)
        self.mutation_controls = [
            self._button(top, 'Add Files…', self.add_files),
            self._button(top, 'Add Folder…', self.add_folder),
            self._button(top, 'Clear', self.clear),
        ]
        self.recursive = QCheckBox(self.tr('Include subfolders'))
        top.addWidget(self.recursive)
        top.addStretch()
        self.scan_button = self._button(top, 'Scan / Rescan', self.scan)
        self.cancel_button = self._button(top, 'Cancel', self.cancel_scan)
        self.cancel_button.setEnabled(False)

        options = QHBoxLayout()
        layout.addLayout(options)
        options.addWidget(QLabel(self.tr('Flag scores below:')))
        self.cutoff = QDoubleSpinBox()
        self.cutoff.setRange(0, 100)
        self.cutoff.setValue(DEFAULT_CUTOFF)
        self.cutoff.setToolTip(self.tr('Star-evidence score; a screening aid, not a stacking weight.'))
        self.cutoff.valueChanged.connect(self.refresh_rows)
        options.addWidget(self.cutoff)
        self.filter = QComboBox()
        self.filter.addItems([self.tr(s) for s in ('All', 'Kept', 'Flagged', 'Errors')])
        self.filter.currentIndexChanged.connect(self.refresh_rows)
        options.addWidget(self.filter)
        self._button(options, 'Select Unflagged', self.select_keepers)
        self._button(options, 'Reset Manual Overrides', self.reset_overrides)
        self.actions_button = self._button(options, 'Actions ▾', self.show_actions)
        self.mutation_controls.append(self.actions_button)
        options.addStretch()

        self.splitter = QSplitter()
        layout.addWidget(self.splitter, 1)
        self.tree = QTreeWidget()
        self.tree.setRootIsDecorated(False)
        self.tree.setHeaderLabels([self.tr(s) for s in (
            'Filename', 'Score', 'Status', 'Reason', 'Stars', 'Coverage %',
            'FWHM (px)', 'Eccentricity', 'Background', 'Noise', 'Exposure (s)', 'Gain', 'Filter', 'Time (ms)',
        )])
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.tree.setSortingEnabled(True)
        self.tree.setColumnWidth(0, 280)
        self.tree.setColumnWidth(3, 240)
        self.tree.currentItemChanged.connect(self._current_changed)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._context_menu)
        self.splitter.addWidget(self.tree)
        self.preview = QLabel(self.tr('Select a frame for an on-demand preview.'))
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(240, 180)
        self.splitter.addWidget(self.preview)
        self.splitter.setSizes([900, 400])
        self.splitter.splitterMoved.connect(lambda *_: self._scale_preview())
        nav = QHBoxLayout()
        layout.addLayout(nav)
        self._button(nav, 'Previous', lambda: self.navigate(-1))
        self._button(nav, 'Next', lambda: self.navigate(1))
        self._button(nav, 'Toggle Flag', self.toggle_flags)
        for label, target in [('Lights', 'lights'), ('Integration', 'integration')]:
            button = self._button(nav, f'Send to Stacking → {label}',
                                  lambda _checked=False, t=target: self.send_stacking(t))
            button.setToolTip(self.tr('Sends selected rows, otherwise all listed files. Use Select Unflagged for keepers.'))
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        self.summary = QLabel()
        layout.addWidget(self.summary)
        self.run_status = QLabel(self.tr('Ready. Scores use raw FITS patches; full images are not loaded during scanning.'))
        self.run_status.setWordWrap(True)
        layout.addWidget(self.run_status)
        self._shortcuts = []
        for key, callback in [('Space', self.toggle_flags), ('Alt+Left', lambda: self.navigate(-1)),
                              ('Alt+Right', lambda: self.navigate(1))]:
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self._shortcuts.append(shortcut)
        self.refresh_rows()

    def _last_folder(self):
        return self.settings.value('ultra_blink/last_folder', '', type=str)

    def add_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, self.tr('Add FITS files'), self._last_folder(),
                                              self.tr('FITS (*.fits *.fit *.fts);;All Files (*)'))
        if paths:
            self.settings.setValue('ultra_blink/last_folder', str(Path(paths[0]).parent))
            self.add_paths(paths)

    def add_folder(self):
        folder = QFileDialog.getExistingDirectory(self, self.tr('Add Folder'), self._last_folder())
        if folder:
            self.settings.setValue('ultra_blink/last_folder', folder)
            try:
                root = Path(folder)
                paths = root.rglob('*') if self.recursive.isChecked() else root.iterdir()
                self.add_paths(sorted(str(p) for p in paths if p.is_file() and
                                      (p.suffix.lower() in FITS_SUFFIXES or
                                       p.name.lower().endswith(('.fits.gz', '.fit.gz', '.fts.gz', '.fz')))))
            except OSError as exc:
                QMessageBox.warning(self, self.tr('Add Folder'), str(exc))

    def add_paths(self, paths):
        if self.scan_thread:
            return
        self.tree.setSortingEnabled(False)
        for path in paths:
            path = str(Path(path).resolve())
            if path in self.frames:
                continue
            self.frames[path] = Frame(path)
            item = SortItem(self.tree)
            item.setData(0, Qt.ItemDataRole.UserRole, path)
            self.items[path] = item
            self._update_row(path)
        self.tree.setSortingEnabled(True)
        self._update_summary()

    def clear(self):
        if self.scan_thread:
            return
        self.preview_target = None
        self.frames.clear()
        self.items.clear()
        self.tree.clear()
        self.preview_cache.clear()
        self.preview_pixmap = None
        self.preview.setText(self.tr('Select a frame for an on-demand preview.'))
        self.progress.setValue(0)
        self._update_summary()

    def scan(self):
        if self.scan_thread or not self.frames:
            return
        for frame in self.frames.values():
            frame.measurement = None
        self.completed = 0
        self.scan_failure = ''
        self.started = perf_counter()
        self.progress.setRange(0, len(self.frames))
        self.progress.setValue(0)
        self.scan_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        for widget in self.mutation_controls:
            widget.setEnabled(False)
        self.run_status.setText(self.tr('Scanning raw FITS patches…'))
        self.refresh_rows()
        thread = ScanThread(list(self.frames), self)
        self.scan_thread = thread
        thread.measured.connect(self._measured)
        thread.failed.connect(self._scan_failed)
        thread.finished.connect(self._scan_finished)
        thread.start()

    def cancel_scan(self):
        if self.scan_thread:
            self.scan_thread.cancel.set()
            self.cancel_button.setEnabled(False)
            self.run_status.setText(self.tr('Cancelling after the current reads…'))

    def _scan_failed(self, message):
        self.scan_failure = message

    def _measured(self, measurement):
        frame = self.frames.get(measurement.path)
        if frame:
            frame.measurement = measurement
            self._update_row(frame.path)
        self.completed += 1
        self.progress.setValue(self.completed)
        self._update_summary()

    def _scan_finished(self):
        thread = self.scan_thread
        cancelled = thread.cancel.is_set()
        self.scan_thread = None
        thread.deleteLater()
        self.scan_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        for widget in self.mutation_controls:
            widget.setEnabled(True)
        message = 'Cancelled' if cancelled else 'Finished'
        if self.scan_failure:
            message = f'Scan failed: {self.scan_failure}'
        self.run_status.setText(f'{message}: {self.completed}/{len(self.frames)} files in '
                                f'{perf_counter()-self.started:.2f} s. Unmeasured files remain pending.')
        self._maybe_close()

    def _update_row(self, path):
        frame = self.frames[path]
        item = self.items[path]
        m = frame.measurement
        flagged = frame.flagged(self.cutoff.value())
        if m is None:
            status, reason = 'Pending', 'Not measured'
        elif m.status != 'ok':
            status, reason = 'Error', m.error
        else:
            status = 'Flagged' if flagged else 'Kept'
            reason = m.rejection_reason(self.cutoff.value())
        if frame.override is not None:
            reason = ('Manual reject' if frame.override else 'Manual keep') + (f'; {reason}' if reason else '')
        values = [Path(path).name, m.score if m and m.status == 'ok' else None, status, reason,
                  m.star_count if m and m.status == 'ok' else None,
                  100*m.coverage if m and m.status == 'ok' else None,
                  m.fwhm if m else None, m.eccentricity if m else None,
                  m.background if m else None, m.noise if m else None,
                  m.exposure if m else '', m.gain if m else '', m.filter_name if m else '',
                  m.seconds*1000 if m else None]
        for col, value in enumerate(values):
            if isinstance(value, float):
                if not math.isfinite(value):
                    value = None
            display = '' if value is None else f'{value:.2f}' if isinstance(value, float) else str(value)
            item.setText(col, display)
            # Tuple prevents missing values from comparing against numeric values.
            numeric = col in (1, 4, 5, 6, 7, 8, 9, 10, 11, 13)
            if numeric:
                try:
                    number = float(value)
                    key = (0, number) if math.isfinite(number) else (1, 0.0)
                except (TypeError, ValueError):
                    key = (1, 0.0)
            else:
                key = str(value or '').casefold()
            item.setData(col, Qt.ItemDataRole.UserRole + 1, key)
        item.setToolTip(0, path)
        item.setToolTip(3, reason)
        item.setForeground(2, QColor('#d98632') if flagged else QColor('#c94e4e') if status == 'Error' else QColor('#6ba477'))
        mode = self.filter.currentIndex()
        visible = (mode == 0 or mode == 1 and frame.keeper(self.cutoff.value()) or
                   mode == 2 and flagged or mode == 3 and m is not None and m.status != 'ok')
        item.setHidden(not visible)
        if not visible:
            item.setSelected(False)

    def refresh_rows(self, *_):
        self.tree.setSortingEnabled(False)
        for path in self.frames:
            self._update_row(path)
        self.tree.setSortingEnabled(True)
        self._update_summary()

    def _update_summary(self):
        frames = list(self.frames.values())
        cutoff = self.cutoff.value()
        kept = sum(f.keeper(cutoff) for f in frames)
        flagged = sum(f.flagged(cutoff) for f in frames)
        errors = sum(f.measurement is not None and f.measurement.status != 'ok' for f in frames)
        pending = sum(f.measurement is None for f in frames)
        self.summary.setText(f'{len(frames)} files • {kept} kept • {flagged} flagged • {errors} errors • {pending} pending')

    def selected_paths(self):
        return [i.data(0, Qt.ItemDataRole.UserRole) for i in self.tree.selectedItems() if not i.isHidden()]

    def flagged_paths(self):
        return [p for p, f in self.frames.items() if f.flagged(self.cutoff.value())]

    def select_keepers(self):
        self.tree.clearSelection()
        if self.filter.currentIndex() not in (0, 1):
            self.filter.setCurrentIndex(1)
        for path, frame in self.frames.items():
            if frame.keeper(self.cutoff.value()):
                self.items[path].setSelected(True)

    def toggle_flags(self):
        for path in self.selected_paths():
            frame = self.frames[path]
            frame.override = not frame.flagged(self.cutoff.value())
        self.refresh_rows()

    def reset_overrides(self):
        for frame in self.frames.values():
            frame.override = None
        self.refresh_rows()

    def clear_flags(self):
        for frame in self.frames.values():
            frame.override = False
        self.refresh_rows()

    def send_stacking(self, target):
        paths = self.selected_paths() or list(self.frames)
        if not paths:
            return
        # Same contract as Blink: selected rows, otherwise every listed file.
        self.sendToStacking.emit(paths, target)
        self.run_status.setText(f'Sent {len(paths)} paths to Stacking → {target}.')

    def navigate(self, direction):
        visible = [self.tree.topLevelItem(i) for i in range(self.tree.topLevelItemCount())
                   if not self.tree.topLevelItem(i).isHidden()]
        if not visible:
            return
        current = self.tree.currentItem()
        index = visible.index(current) if current in visible else (-1 if direction > 0 else len(visible))
        index = min(max(index+direction, 0), len(visible)-1)
        self.tree.clearSelection()
        self.tree.setCurrentItem(visible[index])
        visible[index].setSelected(True)
        self.tree.scrollToItem(visible[index])

    def _current_changed(self, current, _previous):
        self.preview_target = current.data(0, Qt.ItemDataRole.UserRole) if current else None
        self.preview_pixmap = None
        if not self.preview_target:
            self.preview.setText(self.tr('Select a frame for an on-demand preview.'))
            return
        if self.preview_target in self.preview_cache:
            self.preview_cache.move_to_end(self.preview_target)
            self._display_preview(self.preview_cache[self.preview_target])
        else:
            self.preview.setText(self.tr('Loading preview…'))
            self._start_preview()

    def _start_preview(self):
        if self.preview_thread or not self.preview_target or self.closing:
            return
        thread = PreviewThread(self.preview_target, self)
        self.preview_thread = thread
        thread.ready.connect(self._preview_ready)
        thread.finished.connect(self._preview_finished)
        thread.start()

    def _preview_ready(self, path, pixels, error):
        if self.closing or path not in self.frames:
            return
        if pixels is not None:
            image = QImage(pixels.data, pixels.shape[1], pixels.shape[0], pixels.strides[0],
                           QImage.Format.Format_Grayscale8).copy()
            self.preview_cache[path] = image
            self.preview_cache.move_to_end(path)
            while len(self.preview_cache) > 3:
                self.preview_cache.popitem(last=False)
        if path == self.preview_target:
            if error:
                self.preview.setText(f'Preview failed: {error}')
            else:
                self._display_preview(self.preview_cache[path])

    def _preview_finished(self):
        path = self.preview_thread.path
        self.preview_thread.deleteLater()
        self.preview_thread = None
        if self.preview_target and self.preview_target != path and self.preview_target not in self.preview_cache:
            self._start_preview()
        self._maybe_close()

    def _display_preview(self, image):
        self.preview_pixmap = QPixmap.fromImage(image)
        self._scale_preview()

    def _scale_preview(self):
        if self.preview_pixmap:
            self.preview.setPixmap(self.preview_pixmap.scaled(self.preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._scale_preview()

    def _context_menu(self, position):
        item = self.tree.itemAt(position)
        if item and not item.isSelected():
            self.tree.clearSelection()
            self.tree.setCurrentItem(item)
            item.setSelected(True)
        self.show_actions(self.tree.viewport().mapToGlobal(position))

    def show_actions(self, position=None):
        menu = QMenu(self)
        for text, callback in [
            ('Toggle Flag', self.toggle_flags), ('Clear Flags', self.clear_flags),
            ('Reset Manual Overrides', self.reset_overrides),
            ('Open in Document Window', self.open_document),
            ('Rename…', self.rename_one),
            ('Batch Rename Selected…', lambda: self.rename_batch(False)),
            ('Rename Flagged Images…', lambda: self.rename_batch(True)),
            ('Move Selected Items…', lambda: self.move_files(False)),
            ('Move All Flagged Images…', lambda: self.move_files(True)),
            ('Delete Selected Items…', lambda: self.delete_files(False)),
            ('Delete All Flagged Images…', lambda: self.delete_files(True)),
            ('Remove Selected from List', self.remove_selected),
            ('Send to Stacking → Lights', lambda: self.send_stacking('lights')),
            ('Send to Stacking → Integration', lambda: self.send_stacking('integration')),
        ]:
            action = menu.addAction(self.tr(text))
            action.triggered.connect(callback)
            if self.scan_thread and text.startswith(('Rename', 'Batch Rename', 'Move', 'Delete', 'Remove')):
                action.setEnabled(False)
        if not position or isinstance(position, bool):
            position = self.actions_button.mapToGlobal(self.actions_button.rect().bottomLeft())
        menu.exec(position)

    def open_document(self):
        paths = self.selected_paths()
        if not paths or not self.doc_manager:
            return
        try:
            self.doc_manager.open_path(paths[0], open_auxiliary_images=False)
        except Exception as exc:
            QMessageBox.warning(self, self.tr('Open Document'), str(exc))

    def _file_busy(self):
        # On Windows a preview mapping can prevent a rename or deletion.
        if self.scan_thread or self.preview_thread:
            self.run_status.setText(self.tr('Wait for the active scan or preview read before changing files.'))
            return True
        return False

    def remove_selected(self):
        if not self._file_busy():
            self._remove_paths(self.selected_paths())

    def _remove_paths(self, paths):
        for path in paths:
            self.frames.pop(path, None)
            item = self.items.pop(path, None)
            if item:
                self.tree.takeTopLevelItem(self.tree.indexOfTopLevelItem(item))
            self.preview_cache.pop(path, None)
        self._update_summary()

    def rename_one(self):
        if self._file_busy():
            return
        paths = self.selected_paths()
        if not paths:
            return
        path = paths[0]
        name, accepted = QInputDialog.getText(self, self.tr('Rename'), self.tr('New filename:'),
                                             text=Path(path).name)
        if accepted:
            self._rename_pairs([(path, name)])

    def rename_batch(self, flagged):
        if self._file_busy():
            return
        paths = self.flagged_paths() if flagged else self.selected_paths()
        if not paths:
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(self.tr('Rename Flagged Images' if flagged else 'Batch Rename Selected'))
        form = QFormLayout(dialog)
        prefix, suffix = QLineEdit('Bad_' if flagged else ''), QLineEdit()
        form.addRow(self.tr('Prefix:'), prefix)
        if not flagged:
            form.addRow(self.tr('Suffix (before extension):'), suffix)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._rename_pairs([(p, prefix.text()+Path(p).stem+suffix.text()+Path(p).suffix) for p in paths])

    def _rename_pairs(self, pairs):
        failures = []
        for path, name in pairs:
            try:
                if not name or name in ('.', '..') or '/' in name or '\\' in name or '\x00' in name:
                    raise ValueError('Enter a filename without directory separators')
                target = str(Path(path).with_name(name))
                if target == path:
                    continue
                if os.path.lexists(target):
                    raise FileExistsError('Target already exists')
                move_without_overwrite(path, target)
                frame = self.frames.pop(path)
                frame.path = target
                if frame.measurement:
                    frame.measurement = replace(frame.measurement, path=target)
                self.frames[target] = frame
                item = self.items.pop(path)
                item.setData(0, Qt.ItemDataRole.UserRole, target)
                self.items[target] = item
                self.preview_cache.pop(path, None)
                if self.preview_target == path:
                    self.preview_target = target
                self._update_row(target)
            except Exception as exc:
                failures.append(f'{Path(path).name}: {exc}')
        self._operation_result('Rename', len(pairs)-len(failures), failures)

    def move_files(self, flagged):
        if self._file_busy():
            return
        paths = self.flagged_paths() if flagged else self.selected_paths()
        if not paths:
            return
        destination = QFileDialog.getExistingDirectory(self, self.tr('Select Destination Folder'), self._last_folder())
        if not destination:
            return
        moved, failures = [], []
        for path in paths:
            try:
                target = str(Path(destination) / Path(path).name)
                if os.path.lexists(target):
                    raise FileExistsError('Target already exists')
                move_without_overwrite(path, target)
                moved.append(path)
            except Exception as exc:
                failures.append(f'{Path(path).name}: {exc}')
        self._remove_paths(moved)
        self._operation_result('Move', len(moved), failures)

    def delete_files(self, flagged):
        if self._file_busy():
            return
        paths = self.flagged_paths() if flagged else self.selected_paths()
        if not paths:
            return
        answer = QMessageBox.question(self, self.tr('Confirm Deletion'),
            self.tr('Permanently delete {0} {1} files? This action is irreversible.').format(
                len(paths), 'flagged' if flagged else 'selected'),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No, QMessageBox.StandardButton.No)
        if answer != QMessageBox.StandardButton.Yes:
            return
        deleted, failures = [], []
        for path in paths:
            try:
                os.remove(path)
                deleted.append(path)
            except Exception as exc:
                failures.append(f'{Path(path).name}: {exc}')
        self._remove_paths(deleted)
        self._operation_result('Delete', len(deleted), failures)

    def _operation_result(self, operation, count, failures):
        self.run_status.setText(f'{operation}: {count} succeeded, {len(failures)} failed.')
        if failures:
            QMessageBox.warning(self, self.tr(operation), '\n'.join(failures[:10]))

    def _maybe_close(self):
        if self.closing and not self.scan_thread and not self.preview_thread:
            self.close()

    def shutdown_workers(self):
        # Application shutdown can destroy parented windows even when their
        # closeEvent was deferred. Join readers before Qt deletes QThreads.
        self.closing = True
        if self.scan_thread:
            self.scan_thread.cancel.set()
            self.scan_thread.wait()
        if self.preview_thread:
            self.preview_thread.wait()

    def closeEvent(self, event):
        if self.scan_thread or self.preview_thread:
            self.closing = True
            self.cancel_scan()
            self.setEnabled(False)
            event.ignore()
            return
        self.settings.setValue('ultra_blink/geometry', self.saveGeometry())
        super().closeEvent(event)
