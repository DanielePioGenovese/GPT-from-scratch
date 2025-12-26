from pathlib import Path

def load_checkpoint(checkpoint_name, path):
    folder = Path(path)

    if checkpoint_name is None:
        return False

    if folder.exists() and folder.is_dir():
        files = list(folder.glob('step_*.pth'))
        if files:
            if checkpoint_name == 'last':    
                last_checkpoint = max(
                    files,
                    key=lambda p: int(p.stem.split('_')[1]),
                    default=None)
                return last_checkpoint.name if last_checkpoint else False
            else:
                full_path = folder / checkpoint_name
                if full_path.exists() and full_path.is_file():
                    return full_path.name
                else:
                    print(f'No checkpoint "{checkpoint_name}" found!')
                    return False
        else:
            print('Any checkpoint saved')
            return False
    else:
        print('The folder does not exist')
        return False

