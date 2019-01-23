#Omid55
from configparser import SafeConfigParser
import os

class MyConfig():
	def __init__(self):
		self.file_path = 'src/config.ini'

		if os.path.exists(self.file_path):
			self.config = SafeConfigParser()
			self.config.read(self.file_path)
		else:
			self.config = SafeConfigParser()
			self.config.add_section('main')
			self.config.set('main', 'CHATS_THRESHOLD', '-1')  # means not defined
			self.config.set('main', 'BROADCAST_THRESHOLD', '-1')  # means not defined
			with open(self.file_path, 'w') as f:
				self.config.write(f)

	def set(self, name, value):
		self.config.set('main', name, value)
		with open(file_path, 'w') as f:
			self.config.write(f)

	def get(self, name):
		return self.config.get('main', name)

