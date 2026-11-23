# Download FalcoCV

FalcoCV is a modular industrial machine vision environment: workflow-based
inspection pipelines, 2D/3D vision, subpixel metrology, shape matching, depth
processing and a web HMI. It runs entirely on the user's own hardware.

## Standalone bundle

The standalone bundle ships Python, Qt, OpenCV, ONNX Runtime, the native kernel
and the models, so no Python installation is required.

| Platform | Package |
| --- | --- |
| Windows 10/11 x64 | `FalcoCV-1.3.2-windows-x64-setup.exe` (installer) or `FalcoCV-1.3.2-windows-x64.zip` (portable) |
| Linux x86_64 | `FalcoCV-1.3.2-linux-x86_64.AppImage` or `FalcoCV-1.3.2-linux-x86_64.zip` |
| Python 3.13 | `pip install falcocv` (wheel and sdist are attached to the release) |

Download the latest release from
[GitHub Releases](https://github.com/jaesc/falcocv/releases/latest).

## Verify the download

Every release includes `SHA256SUMS.txt`. The Linux AppImage and the checksum
file are signed with the FalcoCV release GPG key:

```text
Fingerprint: 3F5717C512E646A37E22D6B7B2BE6A6DB85C6837
Key:         docs/FALCOCV_GPG_KEY.asc
```

Linux / macOS:

```bash
sha256sum -c SHA256SUMS.txt
gpg --import FALCOCV_GPG_KEY.asc
gpg --verify SHA256SUMS.txt.asc SHA256SUMS.txt
gpg --verify FalcoCV-1.3.2-linux-x86_64.AppImage.asc FalcoCV-1.3.2-linux-x86_64.AppImage
```

Windows PowerShell:

```powershell
Get-FileHash .\FalcoCV-1.3.2-windows-x64-setup.exe -Algorithm SHA256
```

Compare the hash with the matching line in `SHA256SUMS.txt`.

## Code signing

Free code signing provided by [SignPath.io](https://about.signpath.io),
certificate by [SignPath Foundation](https://signpath.org). Windows
Authenticode signing is enabled once the project is approved by SignPath
Foundation; until then, verify the SHA-256 checksums above. Linux release
assets are signed with the FalcoCV GPG key.

See the [code signing policy](../README.md#code-signing-policy) and the
[privacy policy](PRIVACY.md).
