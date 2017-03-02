-- The organization of the project is largely borrowed from 'https://github.com/anewell/pose-hg-train'
-- Thanks for the great work of Alejandro Newell on Human Pose Estimation!


-- Initialization
require 'paths'
paths.dofile('ref.lua') -- Do global variable initialization and parse the command line input
print(projectDir)