; Values supplied by scripts/release.py.
[Setup]
AppId=com.vaeproject.vaesynth
AppName=VAE Synth
AppVersion={#ProductVersion}
AppPublisher=VAEProject
DefaultDirName={autopf}\VAE Synth
MinVersion=10.0.19041
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
Source: "{#Redist}"; DestDir: "{tmp}"; DestName: "vc_redist.x64.exe"; Flags: dontcopy
Source: "{#Artifacts}\Standalone\*"; DestDir: "{app}"; Excludes: "*.lib,*.exp,*.pdb,*.ilk"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Artifacts}\VST3\VAE Synth.vst3\*"; DestDir: "{commoncf64}\VST3\VAE Synth.vst3"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "{#Notices}\*"; DestDir: "{app}\Licenses"; Flags: ignoreversion recursesubdirs createallsubdirs
#if Int(IncludeModels) != 0
Source: "{#Models}\*"; DestDir: "{commonappdata}\VAE Synth\models"; Flags: ignoreversion recursesubdirs createallsubdirs
#endif

[Icons]
Name: "{group}\VAE Synth"; Filename: "{app}\VAE Synth.exe"

[Code]
var
  RuntimeNeedsRestart: Boolean;

function HasCurrentRuntime: Boolean;
var
  Installed, Major, Minor, Build, Revision: Cardinal;
  Key: String;
begin
  Key := 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64';
  Result := False;
  if not RegQueryDWordValue(HKLM64, Key, 'Installed', Installed) or (Installed <> 1) then Exit;
  if not RegQueryDWordValue(HKLM64, Key, 'Major', Major) then Exit;
  if not RegQueryDWordValue(HKLM64, Key, 'Minor', Minor) then Exit;
  if not RegQueryDWordValue(HKLM64, Key, 'Bld', Build) then Exit;
  if not RegQueryDWordValue(HKLM64, Key, 'Rbld', Revision) then Exit;
  Result := (Major > 14) or ((Major = 14) and ((Minor > 44) or
    ((Minor = 44) and (Build >= 35211))));
end;

function PrepareToInstall(var NeedsRestart: Boolean): String;
var
  ExitCode: Integer;
begin
  Result := '';
  if HasCurrentRuntime then Exit;
  ExtractTemporaryFile('vc_redist.x64.exe');
  if not Exec(ExpandConstant('{tmp}\vc_redist.x64.exe'),
    '/install /quiet /norestart', '', SW_HIDE, ewWaitUntilTerminated, ExitCode) then
  begin
    Result := 'Unable to start the Microsoft Visual C++ x64 runtime installer.';
    Exit;
  end;
  if ExitCode = 3010 then RuntimeNeedsRestart := True
  else if (ExitCode <> 0) and not ((ExitCode = 1638) and HasCurrentRuntime) then
    Result := Format('Microsoft Visual C++ x64 runtime installation failed (code %d). See the runtime log in %%TEMP%%.', [ExitCode]);
end;

function NeedRestart: Boolean;
begin
  Result := RuntimeNeedsRestart;
end;
