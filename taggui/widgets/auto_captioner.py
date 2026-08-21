import sys

from PySide6.QtCore import QModelIndex, Qt, Signal, Slot
from PySide6.QtGui import QFontMetrics, QTextCursor
from PySide6.QtWidgets import (QAbstractScrollArea, QDockWidget, QFormLayout,
                               QFrame, QHBoxLayout, QLabel, QMessageBox,
                               QPlainTextEdit, QProgressBar, QScrollArea,
                               QVBoxLayout, QWidget)

from auto_captioning.captioning_thread import CaptioningThread
from dialogs.caption_multiple_images_dialog import CaptionMultipleImagesDialog
from models.image_list_model import ImageListModel
from utils.big_widgets import TallPushButton
from utils.enums import CaptionPosition
from utils.settings import get_settings, get_tag_separator
from utils.settings_widgets import (FocusedScrollSettingsComboBox,
                                    FocusedScrollSettingsDoubleSpinBox,
                                    FocusedScrollSettingsSpinBox,
                                    SettingsBigCheckBox, SettingsLineEdit,
                                    SettingsPlainTextEdit)
from utils.utils import pluralize
from widgets.image_list import ImageList


def set_text_edit_height(text_edit: QPlainTextEdit, line_count: int):
    """
    Set the height of a text edit to the height of a given number of lines.
    """
    # From https://stackoverflow.com/a/46997337.
    document = text_edit.document()
    font_metrics = QFontMetrics(document.defaultFont())
    margins = text_edit.contentsMargins()
    height = int(font_metrics.lineSpacing() * line_count
                 + margins.top() + margins.bottom()
                 + document.documentMargin() * 2
                 + text_edit.frameWidth() * 2)
    text_edit.setFixedHeight(height)


class HorizontalLine(QFrame):
    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.Shape.HLine)
        self.setFrameShadow(QFrame.Shadow.Raised)


class CaptionSettingsForm(QVBoxLayout):
    def __init__(self):
        super().__init__()
        basic_settings_form = QFormLayout()
        basic_settings_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        basic_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.model_line_edit = SettingsLineEdit(key='model_id')
        self.model_line_edit.setPlaceholderText('qwen2.5-vl-7b-instruct')
        self.model_line_edit.setClearButtonEnabled(True)
        self.prompt_text_edit = SettingsPlainTextEdit(key='prompt')
        set_text_edit_height(self.prompt_text_edit, 4)
        self.caption_start_line_edit = SettingsLineEdit(key='caption_start')
        self.caption_start_line_edit.setClearButtonEnabled(True)
        self.caption_position_combo_box = FocusedScrollSettingsComboBox(
            key='caption_position')
        self.caption_position_combo_box.addItems(list(CaptionPosition))
        self.remove_tag_separators_container = QWidget()
        remove_tag_separators_layout = QHBoxLayout(
            self.remove_tag_separators_container)
        remove_tag_separators_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        remove_tag_separators_layout.setContentsMargins(0, 0, 0, 0)
        self.remove_tag_separators_check_box = SettingsBigCheckBox(
            key='remove_tag_separators', default=True)
        remove_tag_separators_label = QLabel(
            'Remove tag separators in caption')
        remove_tag_separators_layout.addWidget(remove_tag_separators_label)
        remove_tag_separators_layout.addWidget(
            self.remove_tag_separators_check_box)
        basic_settings_form.addRow('Model', self.model_line_edit)
        basic_settings_form.addRow('Prompt', self.prompt_text_edit)
        basic_settings_form.addRow('Start caption with',
                                   self.caption_start_line_edit)
        basic_settings_form.addRow('Caption position',
                                   self.caption_position_combo_box)
        basic_settings_form.addRow(self.remove_tag_separators_container)

        self.toggle_advanced_settings_form_button = TallPushButton(
            'Show Advanced Settings')

        self.advanced_settings_form_container = QWidget()
        advanced_settings_form = QFormLayout(
            self.advanced_settings_form_container)
        advanced_settings_form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        advanced_settings_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        bad_forced_words_form = QFormLayout()
        bad_forced_words_form.setRowWrapPolicy(
            QFormLayout.RowWrapPolicy.WrapAllRows)
        bad_forced_words_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        self.bad_words_line_edit = SettingsLineEdit(key='bad_words')
        self.bad_words_line_edit.setClearButtonEnabled(True)
        self.forced_words_line_edit = SettingsLineEdit(key='forced_words')
        self.forced_words_line_edit.setClearButtonEnabled(True)
        self.tag_source_line_edit = SettingsLineEdit(key='tag_source')
        self.tag_source_line_edit.setClearButtonEnabled(True)
        bad_forced_words_form.addRow('Discourage from caption',
                                     self.bad_words_line_edit)
        bad_forced_words_form.addRow('Include in caption',
                                     self.forced_words_line_edit)
        bad_forced_words_form.addRow('Tag source directory',
                                     self.tag_source_line_edit)
        self.min_new_token_count_spin_box = FocusedScrollSettingsSpinBox(
            key='min_new_tokens', default=1, minimum=1, maximum=999)
        self.max_new_token_count_spin_box = FocusedScrollSettingsSpinBox(
            key='max_new_tokens', default=100, minimum=1, maximum=999)
        self.beam_count_spin_box = FocusedScrollSettingsSpinBox(
            key='num_beams', default=1, minimum=1, maximum=99)
        self.length_penalty_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='length_penalty', default=1, minimum=-5, maximum=5)
        self.length_penalty_spin_box.setSingleStep(0.1)
        self.use_sampling_check_box = SettingsBigCheckBox(key='do_sample',
                                                          default=False)
        # The temperature must be positive.
        self.temperature_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='temperature', default=1, minimum=0.01, maximum=2)
        self.temperature_spin_box.setSingleStep(0.01)
        self.top_k_spin_box = FocusedScrollSettingsSpinBox(
            key='top_k', default=50, minimum=0, maximum=200)
        self.top_p_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='top_p', default=1, minimum=0, maximum=1)
        self.top_p_spin_box.setSingleStep(0.01)
        self.repetition_penalty_spin_box = FocusedScrollSettingsDoubleSpinBox(
            key='repetition_penalty', default=1, minimum=1, maximum=2)
        self.repetition_penalty_spin_box.setSingleStep(0.01)
        self.no_repeat_ngram_size_spin_box = FocusedScrollSettingsSpinBox(
            key='no_repeat_ngram_size', default=3, minimum=0, maximum=5)
        advanced_settings_form.addRow(bad_forced_words_form)
        advanced_settings_form.addRow(HorizontalLine())
        advanced_settings_form.addRow('Minimum tokens',
                                      self.min_new_token_count_spin_box)
        advanced_settings_form.addRow('Maximum tokens',
                                      self.max_new_token_count_spin_box)
        advanced_settings_form.addRow('Number of beams',
                                      self.beam_count_spin_box)
        advanced_settings_form.addRow('Length penalty',
                                      self.length_penalty_spin_box)
        advanced_settings_form.addRow('Use sampling',
                                      self.use_sampling_check_box)
        advanced_settings_form.addRow('Temperature',
                                      self.temperature_spin_box)
        advanced_settings_form.addRow('Top-k', self.top_k_spin_box)
        advanced_settings_form.addRow('Top-p', self.top_p_spin_box)
        advanced_settings_form.addRow('Repetition penalty',
                                      self.repetition_penalty_spin_box)
        advanced_settings_form.addRow('No repeat n-gram size',
                                      self.no_repeat_ngram_size_spin_box)
        self.advanced_settings_form_container.hide()

        self.addLayout(basic_settings_form)
        self.horizontal_line = HorizontalLine()
        self.addWidget(self.horizontal_line)
        self.addWidget(self.toggle_advanced_settings_form_button)
        self.addWidget(self.advanced_settings_form_container)
        self.addStretch()

        self.toggle_advanced_settings_form_button.clicked.connect(
            self.toggle_advanced_settings_form)
        # Make sure the minimum new token count is less than or equal to the
        # maximum new token count.
        self.min_new_token_count_spin_box.valueChanged.connect(
            self.max_new_token_count_spin_box.setMinimum)
        self.max_new_token_count_spin_box.valueChanged.connect(
            self.min_new_token_count_spin_box.setMaximum)

    @Slot()
    def toggle_advanced_settings_form(self):
        if self.advanced_settings_form_container.isHidden():
            self.advanced_settings_form_container.show()
            self.toggle_advanced_settings_form_button.setText(
                'Hide Advanced Settings')
        else:
            self.advanced_settings_form_container.hide()
            self.toggle_advanced_settings_form_button.setText(
                'Show Advanced Settings')

    def get_caption_settings(self) -> dict:
        return {
            'model_id': self.model_line_edit.text(),
            'prompt': self.prompt_text_edit.toPlainText(),
            'caption_start': self.caption_start_line_edit.text(),
            'caption_position': self.caption_position_combo_box.currentText(),
            'remove_tag_separators':
                self.remove_tag_separators_check_box.isChecked(),
            'bad_words': self.bad_words_line_edit.text(),
            'forced_words': self.forced_words_line_edit.text(),
            'tag_source': self.tag_source_line_edit.text(),
            'generation_parameters': {
                'min_new_tokens': self.min_new_token_count_spin_box.value(),
                'max_new_tokens': self.max_new_token_count_spin_box.value(),
                'num_beams': self.beam_count_spin_box.value(),
                'length_penalty': self.length_penalty_spin_box.value(),
                'do_sample': self.use_sampling_check_box.isChecked(),
                'temperature': self.temperature_spin_box.value(),
                'top_k': self.top_k_spin_box.value(),
                'top_p': self.top_p_spin_box.value(),
                'repetition_penalty': self.repetition_penalty_spin_box.value(),
                'no_repeat_ngram_size':
                    self.no_repeat_ngram_size_spin_box.value()
            }
        }


@Slot()
def restore_stdout_and_stderr():
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__


class AutoCaptioner(QDockWidget):
    caption_generated = Signal(QModelIndex, str, list)

    def __init__(self, image_list_model: ImageListModel,
                 image_list: ImageList):
        super().__init__()
        self.image_list_model = image_list_model
        self.image_list = image_list
        self.is_captioning = False
        self.captioning_thread = None
        # Whether the last block of text in the console text edit should be
        # replaced with the next block of text that is outputted.
        self.replace_last_console_text_edit_block = False

        # Each `QDockWidget` needs a unique object name for saving its state.
        self.setObjectName('auto_captioner')
        self.setWindowTitle('Auto-Captioner')
        self.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea
                             | Qt.DockWidgetArea.RightDockWidgetArea)

        self.start_cancel_button = TallPushButton('Start Auto-Captioning')
        self.progress_bar = QProgressBar()
        self.progress_bar.setFormat('%v / %m images captioned (%p%)')
        self.progress_bar.hide()
        self.console_text_edit = QPlainTextEdit()
        set_text_edit_height(self.console_text_edit, 4)
        self.console_text_edit.setReadOnly(True)
        self.console_text_edit.hide()
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.addWidget(self.start_cancel_button)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.console_text_edit)
        self.caption_settings_form = CaptionSettingsForm()
        layout.addLayout(self.caption_settings_form)
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setSizeAdjustPolicy(
            QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidget(container)
        self.setWidget(scroll_area)

        self.start_cancel_button.clicked.connect(
            self.start_or_cancel_captioning)

    @Slot()
    def start_or_cancel_captioning(self):
        if self.is_captioning:
            # Cancel captioning.
            self.captioning_thread.is_canceled = True
            self.start_cancel_button.setEnabled(False)
            self.start_cancel_button.setText('Canceling Auto-Captioning...')
        else:
            # Start captioning.
            self.generate_captions()

    def set_is_captioning(self, is_captioning: bool):
        self.is_captioning = is_captioning
        button_text = ('Cancel Auto-Captioning' if is_captioning
                       else 'Start Auto-Captioning')
        self.start_cancel_button.setText(button_text)

    @Slot(str)
    def update_console_text_edit(self, text: str):
        # '\x1b[A' is the ANSI escape sequence for moving the cursor up.
        if text == '\x1b[A':
            self.replace_last_console_text_edit_block = True
            return
        text = text.strip()
        if not text:
            return
        if self.console_text_edit.isHidden():
            self.console_text_edit.show()
        if self.replace_last_console_text_edit_block:
            self.replace_last_console_text_edit_block = False
            # Select and remove the last block of text.
            self.console_text_edit.moveCursor(QTextCursor.MoveOperation.End)
            self.console_text_edit.moveCursor(
                QTextCursor.MoveOperation.StartOfBlock,
                QTextCursor.MoveMode.KeepAnchor)
            self.console_text_edit.textCursor().removeSelectedText()
            # Delete the newline.
            self.console_text_edit.textCursor().deletePreviousChar()
        self.console_text_edit.appendPlainText(text)

    @Slot()
    def show_alert(self):
        if self.captioning_thread.is_canceled:
            return
        if self.captioning_thread.is_error:
            icon = QMessageBox.Icon.Critical
            text = ('An error occurred during captioning. See the '
                    'Auto-Captioner console for more information.')
        else:
            icon = QMessageBox.Icon.Information
            text = 'Captioning has finished.'
        alert = QMessageBox()
        alert.setIcon(icon)
        alert.setText(text)
        alert.exec()

    @Slot()
    def generate_captions(self):
        selected_image_indices = self.image_list.get_selected_image_indices()
        selected_image_count = len(selected_image_indices)
        show_alert_when_finished = False
        if selected_image_count > 1:
            confirmation_dialog = CaptionMultipleImagesDialog(
                selected_image_count)
            reply = confirmation_dialog.exec()
            if reply != QMessageBox.StandardButton.Yes:
                return
            show_alert_when_finished = (confirmation_dialog
                                        .show_alert_check_box.isChecked())
        self.set_is_captioning(True)
        caption_settings = self.caption_settings_form.get_caption_settings()
        if caption_settings['caption_position'] != CaptionPosition.DO_NOT_ADD:
            self.image_list_model.add_to_undo_stack(
                action_name=f'Generate '
                            f'{pluralize("Caption", selected_image_count)}',
                should_ask_for_confirmation=selected_image_count > 1)
        if selected_image_count > 1:
            self.progress_bar.setRange(0, selected_image_count)
            self.progress_bar.setValue(0)
            self.progress_bar.show()
        settings = get_settings()
        caption_settings['api_base_url'] = settings.value(
            'api_base_url', 'http://127.0.0.1:1234/v1', type=str)
        caption_settings['api_key'] = settings.value('api_key', '', type=str)
        caption_settings['request_timeout'] = settings.value(
            'request_timeout', 120, type=int)
        self.captioning_thread = CaptioningThread(
            self, self.image_list_model, selected_image_indices,
            caption_settings, get_tag_separator())
        self.captioning_thread.text_outputted.connect(
            self.update_console_text_edit)
        self.captioning_thread.clear_console_text_edit_requested.connect(
            self.console_text_edit.clear)
        self.captioning_thread.caption_generated.connect(
            self.caption_generated)
        self.captioning_thread.progress_bar_update_requested.connect(
            self.progress_bar.setValue)
        self.captioning_thread.finished.connect(
            lambda: self.set_is_captioning(False))
        self.captioning_thread.finished.connect(restore_stdout_and_stderr)
        self.captioning_thread.finished.connect(self.progress_bar.hide)
        self.captioning_thread.finished.connect(
            lambda: self.start_cancel_button.setEnabled(True))
        if show_alert_when_finished:
            self.captioning_thread.finished.connect(self.show_alert)
        # Redirect `stdout` and `stderr` so that the outputs are displayed in
        # the console text edit.
        sys.stdout = self.captioning_thread
        sys.stderr = self.captioning_thread
        self.captioning_thread.start()
