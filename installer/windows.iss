; Values supplied by scripts/release.py.
[Setup]
AppId=com.vaeproject.vaesynth
AppName=VAE Synth
AppVersion={#ProductVersion}
AppPublisher=VAEProject
DefaultDirName={autopf}\VAE Synth
DefaultGroupName=VAE Synth
OutputDir={#OutputDir}
OutputBaseFilename=installer
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=admin
Compression=lzma2
SolidCompression=yes
UninstallDisplayIcon={app}\VAE Synth.exe

[Files]
Source: "{#Artifacts}\Standalone\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Artifacts}\VST3\VAE Synth.vst3\*"; DestDir: "{commoncf64}\VST3\VAE Synth.vst3"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Notices}\*"; DestDir: "{app}\Licenses"; Flags: ignoreversion recursesubdirs createallsubdirs

Source: "{#Models}\*"; DestDir: "{commonappdata}\VAE Synth\models"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\VAE Synth"; Filename: "{app}\VAE Synth.exe"
