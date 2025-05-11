#!/bin/bash

# project_context_builder.sh
# Description: Creates a comprehensive project context file for LLM input
# Usage: ./project_context_builder.sh [output_file] [additional_dirs_to_exclude]

set -e  # Exit on any error

# Default output file
OUTPUT_FILE="${1:-project_context.txt}"
# Additional directories to exclude (comma separated)
ADDITIONAL_EXCLUDE="${2:-}"

echo "Project Context Builder"
echo "----------------------"
echo "Creating context file at: $OUTPUT_FILE"

# Function to check if command exists
command_exists() {
  command -v "$1" >/dev/null 2>&1
}

# Create header for the output file
cat > "$OUTPUT_FILE" << EOL
# PROJECT CONTEXT DOCUMENT
# Generated on: $(date)
# Purpose: This document contains the structure and content of the project for LLM context

## TABLE OF CONTENTS
1. Project Structure
2. File Contents

EOL

echo "Capturing project structure..."

# Define directories to exclude (always exclude virtual environments and package directories)
EXCLUDE_DIRS="trading_env,venv,env,.env,virtualenv,node_modules,__pycache__,.git,.idea,.vscode,dist,build"

# Add any additional exclusions
if [ -n "$ADDITIONAL_EXCLUDE" ]; then
  EXCLUDE_DIRS="$EXCLUDE_DIRS,$ADDITIONAL_EXCLUDE"
fi

echo "Excluding directories: $EXCLUDE_DIRS"

# Convert comma-separated list to an array
IFS=',' read -ra EXCLUDE_ARRAY <<< "$EXCLUDE_DIRS"

# Create find exclusion arguments
FIND_EXCLUDE=""
for dir in "${EXCLUDE_ARRAY[@]}"; do
  FIND_EXCLUDE="$FIND_EXCLUDE -not -path '*/$dir/*'"
done

# Create tree exclusion pattern
TREE_EXCLUDE=$(echo "$EXCLUDE_DIRS" | tr ',' '|')

# Get list of all directories to process (excluding the ones we don't want)
DIRS_TO_PROCESS=$(eval "find . -type d -not -path '*/\\.*' $FIND_EXCLUDE" | sort)

echo -e "\n## 1. PROJECT STRUCTURE\n" >> "$OUTPUT_FILE"

# Get the project structure using tree if available, otherwise use find
if command_exists tree; then
  echo "Using 'tree' command to generate structure..."
  echo '```' >> "$OUTPUT_FILE"
  # Use tree with our exclusion pattern
  tree -I "$TREE_EXCLUDE" --dirsfirst . >> "$OUTPUT_FILE" 2>/dev/null || echo "Error running tree, falling back to find..."
  echo '```' >> "$OUTPUT_FILE"
else
  echo "Note: 'tree' command not found, using 'find' command instead." >> "$OUTPUT_FILE"
  echo '```' >> "$OUTPUT_FILE"
  # List directories
  echo "DIRECTORIES:" >> "$OUTPUT_FILE"
  echo "$DIRS_TO_PROCESS" >> "$OUTPUT_FILE"
  echo "" >> "$OUTPUT_FILE"
  
  # List Python, Markdown, and env files not in excluded directories
  echo "FILES:" >> "$OUTPUT_FILE"
  eval "find . -type f \( -name \"*.py\" -o -name \"*.md\" -o -name \".env*\" \) $FIND_EXCLUDE" | sort >> "$OUTPUT_FILE"
  echo '```' >> "$OUTPUT_FILE"
fi

echo -e "\n## 2. FILE CONTENTS\n" >> "$OUTPUT_FILE"

# Find all relevant files, excluding our excluded directories
echo "Collecting file contents..."
FILES=$(eval "find . -type f \( -name \"*.py\" -o -name \"*.md\" -o -name \".env*\" \) $FIND_EXCLUDE" | sort)

# Extra safety check to filter out any remaining virtual environment files
FILES=$(echo "$FILES" | grep -v "trading_env" | grep -v "/venv/" | grep -v "/env/" | grep -v "/\.env/")

# Counter for progress
TOTAL_FILES=$(echo "$FILES" | wc -l)
CURRENT=0

echo "Found $TOTAL_FILES files to process"

# Process each file
for FILE in $FILES; do
  CURRENT=$((CURRENT + 1))
  
  # Skip files that are too large (over 1MB)
  FILE_SIZE=$(wc -c < "$FILE" 2>/dev/null || echo "0")
  if [ "$FILE_SIZE" -gt 1000000 ]; then
    echo "[$CURRENT/$TOTAL_FILES] Skipping large file ($(du -h "$FILE" | cut -f1)): $FILE"
    echo -e "\n### $FILE\n" >> "$OUTPUT_FILE"
    echo "Content omitted - file too large ($(du -h "$FILE" | cut -f1))" >> "$OUTPUT_FILE"
    continue
  fi
  
  # Double-check this isn't in a virtual environment
  if echo "$FILE" | grep -q -E "/(trading_env|venv|env|\.env|virtualenv)/"; then
    echo "[$CURRENT/$TOTAL_FILES] Skipping virtual environment file: $FILE"
    continue
  fi
  
  echo "[$CURRENT/$TOTAL_FILES] Processing: $FILE"
  
  # Add file header with relative path
  echo -e "\n### $FILE\n" >> "$OUTPUT_FILE"
  
  # Add file type and basic info
  FILE_EXT="${FILE##*.}"
  if [[ "$FILE" == *".env"* ]]; then
    FILE_TYPE="Environment Configuration"
  elif [ "$FILE_EXT" == "py" ]; then
    FILE_TYPE="Python Source Code"
  elif [ "$FILE_EXT" == "md" ]; then
    FILE_TYPE="Markdown Documentation"
  else
    FILE_TYPE="Unknown Type"
  fi
  
  echo "Type: $FILE_TYPE" >> "$OUTPUT_FILE"
  echo "Size: $(du -h "$FILE" | cut -f1)" >> "$OUTPUT_FILE"
  
  # Add file content with markdown code block
  echo '```'"$FILE_EXT" >> "$OUTPUT_FILE"
  cat "$FILE" >> "$OUTPUT_FILE" 2>/dev/null || echo "Error reading file"
  echo '```' >> "$OUTPUT_FILE"
done

# Add footer
echo -e "\n## END OF PROJECT CONTEXT DOCUMENT\n" >> "$OUTPUT_FILE"

echo "----------------------"
echo "✅ Context file created successfully at: $OUTPUT_FILE"
echo "File size: $(du -h "$OUTPUT_FILE" | cut -f1)"
echo "Number of files processed: $TOTAL_FILES"
echo ""
echo "You can now provide this file to Claude or other LLMs to give full context about your project."