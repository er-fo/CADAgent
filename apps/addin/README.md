# CADAgent Fusion 360 add-in

Runs inside Fusion 360. Sends requests to the backend over a websocket and applies
the returned CAD operations.

## Layout

| Path | What |
|---|---|
| `mac/CADAgent/` | macOS add-in, with vendored Python dependencies under `lib/`. |
| `win/CADAgent/` | Windows add-in, same contents. |
| `mac/README.md`, `win/README.md` | Per-platform install instructions. |
| `mac/CADAgent/README.md` | Add-in internals and the websocket protocol. |

The two platform trees are currently byte-identical. Keep them in sync when you
change shared code — nothing enforces this automatically.

## Configuration

`CADAgent/.env.cadagent` is tracked and ships in the release zip. It holds only
public values: the Supabase project URL and publishable key, and the backend
websocket host. Point the add-in at your own backend by editing `BACKEND_HOST`.

## Releases

`.github/workflows/release-addin.yml` (manual `workflow_dispatch`) stamps the
version into `CADAgent.manifest` and `PackageContents.xml`, then publishes
`CADAgent-macOS.zip` and `CADAgent-Windows.zip` to a GitHub Release. Each zip has a
single top-level `CADAgent/` directory, which is what users drop into Fusion's
`AddIns` folder.
