#!/bin/bash
echo "📂 Copying wiki files..."
cp -r "/Users/christelle.nollet/Documents/Obsidian Vault/wiki/." ~/Documents/robotics-wiki-site/content/

echo "📝 Committing changes..."
cd ~/Documents/robotics-wiki-site
git add .
git commit -m "Update wiki $(date '+%Y-%m-%d')"

echo "🚀 Pushing to GitHub..."
git push origin v4

echo "✅ Done — site will rebuild in ~2 minutes"
echo "🔗 https://christelle1208.github.io/robotics-wiki-site/"
