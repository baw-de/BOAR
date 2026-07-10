import os

# 1. Replace header in modules.rst
modules_path = "docs/source/modules.rst"
with open(modules_path) as f:
    content = f.read()

content = content.replace("basement-tools\n==============", "API Reference\n=============")
with open(modules_path, "w") as f:
    f.write(content)

# 2. Generate YAML docs and add to user_defined_configs.rst
yaml_files = [f for f in os.listdir("user_defined_configs") if f.endswith(".yaml")]

user_configs_path = "docs/source/user_defined_configs.rst"
with open(user_configs_path) as f:
    user_configs_content = f.read()

for yaml_file in yaml_files:
    name = yaml_file.replace("user_defined_configs.", "").replace(".yaml", "")
    entry = f"   user_defined_configs.{name}\n"

    # Skip if already exists
    if entry in user_configs_content:
        print(f"Already exists: user_defined_configs.{name}")
        continue

    # Create RST file for YAML
    rst_content = f"""{name}
============

.. literalinclude:: ../../user_defined_configs/{yaml_file}
   :language: yaml
"""
    with open(f"docs/source/user_defined_configs.{name}.rst", "w") as f:
        f.write(rst_content)

    # Add to toctree (after last entry, before next section)
    # Find the toctree block and add before "Module contents"
    lines = user_configs_content.split("\n")
    new_content = []
    added = False

    for i, line in enumerate(lines):
        new_content.append(line)
        # Add after last toctree entry, before "Module contents"
        if line.strip() == "Module contents" and not added:
            # Go back and find toctree
            for j in range(len(new_content) - 1, -1, -1):
                if new_content[j].strip() == ".. toctree::":
                    # Add after all entries in this toctree
                    break
            # Insert at correct position
            new_content.insert(-1, entry)
            added = True
            print(f"Added: user_defined_configs.{name}")

    user_configs_content = "\n".join(new_content)

# Write updated content
with open(user_configs_path, "w") as f:
    f.write(user_configs_content)

print(f"Done! Added {len(yaml_files)} YAML docs")
