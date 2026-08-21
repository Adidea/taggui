# TagGUI

Vibecoded fork that removes the outdated built-in model list and adds a openai compatible api connection. Enter your localhost in settings menu

should work

known issues: changing max image tokens from default in koboldcpp makes model blind (raising minimum and raising resolution works fine, untested on other backends)
              reasoning output leaks sometimes into the final caption; avoid using thinking mode on models
