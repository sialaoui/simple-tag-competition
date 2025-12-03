# Setup Instructions

Follow these steps to set up the Simple Tag Competition repository.

## 1. Initial Repository Setup

```bash
# Initialize git repository
cd /path/to/simple-tag-competition
git init
git add .
git commit -m "Initial commit: Simple Tag Competition setup"

# Create repository on GitHub, then:
git remote add origin https://github.com/nathanael-fijalkow/simple-tag-competition.git
git branch -M main
git push -u origin main
```

## 2. Reference Prey (Public) and Student Predator

Students only implement the predator. The course provides a public prey model.

1. Reference prey files (already in repo):
   - `reference_agents_source/prey_agent.py`
   - `reference_agents_source/prey_final_model.zip`

2. Students submit only:
   - `submissions/<username>/agent.py` (predator)
   - optionally `*.pth` files in the same folder

3. The GitHub Actions workflow copies the public prey into `reference_agents/` and evaluates student predator vs prey.

## 3. Configure GitHub Pages

1. Go to repository Settings → Pages
2. Under "Build and deployment":
   - Source: **GitHub Actions**
3. Save

The leaderboard will be available at:
`https://nathanael-fijalkow.github.io/simple-tag-competition/`

Evaluation uses deterministic seeds per episode.

## 5. Test the Setup

### Test Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Test the template
python test_agent.py template/agent.py

# Test evaluation (student predator vs public prey)
python evaluate.py \
  --submission-dir template \
   --reference-agents-dir reference_agents \
  --output results/test.json \
  --episodes 10

# Generate leaderboard
python generate_leaderboard.py
```

Note: The environment uses a discrete action space `Discrete(5)` (0=no action, 1=left, 2=right, 3=down, 4=up). Ensure your agents return integers in `[0, 4]`.

### Test with a Mock PR

1. Create a test submission:
   ```bash
   mkdir -p submissions/test_student
   cp template/agent.py submissions/test_student/
   ```

2. Create a branch and PR:
   ```bash
   git checkout -b test-submission
   git add submissions/test_student/
   git commit -m "Test submission"
   git push origin test-submission
   ```

3. Create a Pull Request on GitHub

4. Watch the Actions tab to see evaluation run

5. Check that:
   - Evaluation runs successfully
   - Results are commented on PR
   - Leaderboard updates
   - GitHub Pages deploys

## 6. Enable PR Evaluation

Make sure GitHub Actions have write permissions:

1. Go to Settings → Actions → General
2. Under "Workflow permissions":
   - Select "Read and write permissions"
   - Check "Allow GitHub Actions to create and approve pull requests"
3. Save

## 7. Protect Main Branch (Optional)

1. Go to Settings → Branches
2. Add branch protection rule for `main`:
   - Require pull request reviews
   - Require status checks to pass
   - Do not require reviews for evaluation bot

## 8. Create Initial Leaderboard

Generate an empty leaderboard:

```bash
python generate_leaderboard.py
git add docs/index.html
git commit -m "Initialize empty leaderboard"
git push
```

## 9. Announce to Students

Share with your students:
- Repository URL
- Leaderboard URL: `https://yourusername.github.io/simple-tag-competition/`
- Instructions from README.md
- Template at `template/agent.py`

## Troubleshooting

### Private Agents Not Found in Workflow

- Check that secrets are properly set
- Verify base64 encoding/decoding
- Check workflow logs for errors

### Leaderboard Not Updating

- Verify GitHub Pages is enabled
- Check that `docs/` directory is being committed
- Look at deploy workflow logs

### Evaluation Failing

- Test locally first
- Check Python version compatibility
- Verify all dependencies are in requirements.txt
- Review workflow logs in Actions tab

## Security Checklist

- [ ] Private agents are NOT in public repository
- [ ] Private agents are stored as GitHub Secrets
- [ ] `.gitignore` includes `private_agents/`
<!-- Seeding removed from evaluation pipeline -->
- [ ] Model weights (if any) are also kept private
- [ ] Workflow has appropriate permissions

## Maintenance

### Updating Private Agents

When you improve your private agents:

1. Update the secrets:
   ```bash
   base64 -i new_prey_agent.py | pbcopy  # macOS
   # Paste into GitHub Secrets
   ```

2. Test locally first
3. Previous submissions will maintain their old scores

### Clearing the Leaderboard

To start fresh:

```bash
rm results/*.json
python generate_leaderboard.py
git add results/ docs/
git commit -m "Reset leaderboard"
git push
```

## Support

For issues:
1. Check workflow logs in Actions tab
2. Test scripts locally
3. Review this setup guide
4. Check GitHub Actions documentation

Good luck with your RL class! 
