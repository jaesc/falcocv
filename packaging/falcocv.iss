; FalcoCV standalone installer (Inno Setup 6).
;
; Built by scripts/build_standalone.py --installer, which passes:
;   /DAppVersion=<version> /DSourceDir=<dist\standalone\FalcoCV>
;   /DOutputDir=<dist\standalone>
;
; Recipient association is per-user by default; the user can choose a
; machine-wide install from the privilege dialog.

#define AppName "FalcoCV"
#define AppPublisher "FalcoCV contributors"
#define AppExe "FalcoCV.exe"
#define HmiExe "FalcoCVHMI.exe"
#define ServerExe "FalcoCVServer.exe"
#define CliExe "FalcoCVCLI.exe"
#define IconFile "..\src\falcocv\ui\assets\icon.ico"

#ifndef AppVersion
  #define AppVersion "0.0.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\dist\standalone\FalcoCV"
#endif
#ifndef OutputDir
  #define OutputDir "..\dist\standalone"
#endif

[Setup]
AppId={{8F3C1D2A-5B7E-4C9A-9E21-7D6B4F0A3C55}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
AppVerName={#AppName} {#AppVersion}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog commandline
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
ChangesAssociations=yes
CloseApplications=yes
RestartApplications=no
Compression=lzma2/fast
SolidCompression=no
OutputDir={#OutputDir}
OutputBaseFilename=FalcoCV-{#AppVersion}-windows-x64-setup
SetupIconFile={#IconFile}
UninstallDisplayIcon={app}\{#AppExe}
UninstallDisplayName={#AppName} {#AppVersion}
WizardStyle=modern
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "spanish"; MessagesFile: "compiler:Languages\Spanish.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppExe}"
Name: "{group}\{#AppName} HMI Designer"; Filename: "{app}\{#HmiExe}"
Name: "{group}\{#AppName} Server"; Filename: "{app}\{#ServerExe}"
Name: "{group}\{cm:UninstallProgram,{#AppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExe}"; Tasks: desktopicon

[Registry]
; .cvj -> Vision Editor
Root: HKA; Subkey: "Software\Classes\.cvj"; ValueType: string; ValueName: ""; ValueData: "FalcoCV.Job"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\FalcoCV.Job"; ValueType: string; ValueName: ""; ValueData: "FalcoCV Job"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\FalcoCV.Job\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#AppExe},0"
Root: HKA; Subkey: "Software\Classes\FalcoCV.Job\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#AppExe}"" ""%1"""

; .hmi -> HMI Designer
Root: HKA; Subkey: "Software\Classes\.hmi"; ValueType: string; ValueName: ""; ValueData: "FalcoCV.HMI"; Flags: uninsdeletevalue
Root: HKA; Subkey: "Software\Classes\FalcoCV.HMI"; ValueType: string; ValueName: ""; ValueData: "FalcoCV HMI Project"; Flags: uninsdeletekey
Root: HKA; Subkey: "Software\Classes\FalcoCV.HMI\DefaultIcon"; ValueType: string; ValueName: ""; ValueData: "{app}\{#HmiExe},0"
Root: HKA; Subkey: "Software\Classes\FalcoCV.HMI\shell\open\command"; ValueType: string; ValueName: ""; ValueData: """{app}\{#HmiExe}"" ""%1"""

[Run]
Filename: "{app}\{#AppExe}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The installer owns the bundle; user data stays in %APPDATA%\FalcoCV on purpose.
Type: filesandordirs; Name: "{app}\site-packages"
