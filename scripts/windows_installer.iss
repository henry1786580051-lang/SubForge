#ifndef MyAppVersion
  #define MyAppVersion "0.0.0-dev"
#endif
#ifndef MyAppSuffix
  #define MyAppSuffix ""
#endif
#ifndef MyCompression
  #define MyCompression "lzma2/ultra64"
#endif

[Setup]
AppId={{D86D8B72-A76B-4F5D-BF36-B6781971C703}
AppName=SubForge
AppVersion={#MyAppVersion}
AppPublisher=SubForge
AppPublisherURL=https://github.com/henry1786580051-lang/SubForge
DefaultDirName={autopf}\SubForge
DefaultGroupName=SubForge
DisableProgramGroupPage=yes
OutputDir=..\artifacts
OutputBaseFilename=SubForge-{#MyAppVersion}-windows-x64{#MyAppSuffix}-setup
Compression={#MyCompression}
SolidCompression=yes
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
WizardStyle=modern
UninstallDisplayIcon={app}\SubForge.exe
SetupLogging=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "..\dist\SubForge\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\SubForge"; Filename: "{app}\SubForge.exe"
Name: "{autodesktop}\SubForge"; Filename: "{app}\SubForge.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\SubForge.exe"; Description: "{cm:LaunchProgram,SubForge}"; Flags: nowait postinstall skipifsilent
