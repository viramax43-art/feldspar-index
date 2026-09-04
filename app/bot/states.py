"""FSM states for Telegram bot."""

from aiogram.fsm.state import State, StatesGroup


class VoiceCollection(StatesGroup):
    collecting = State()


class SpeakMode(StatesGroup):
    waiting_text = State()
