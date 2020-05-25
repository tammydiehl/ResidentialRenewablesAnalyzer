# -*- mode: python -*-

block_cipher = None


a = Analysis(['SVETapp.py'],
             pathex=['C:\\Users\\ptng009\\Documents\\Gitlab\\dvet-stage\\dervet\\storagevet'],
             binaries=[],
             datas=[('Images','.\\Images\\'),('Schema.xml','.'),('Datasets','.\\Datasets\\')],
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=['PyQt5', 'cvxopt'],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher,
             noarchive=False)
pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)
exe = EXE(pyz,
          a.scripts,
          a.binaries,
          a.zipfiles,
          a.datas,
          [],
          name='SVETapp',
          debug=False,
          bootloader_ignore_signals=False,
          strip=False,
          upx=True,
          runtime_tmpdir=None,
          onedir=True,
          console=True , icon='Images\\favicon.ico')
coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=True,
               name='SVETapp')

