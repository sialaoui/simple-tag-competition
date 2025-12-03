# Simple Tag Competition - Launch Checklist

Use this checklist to ensure everything is set up correctly before launching your competition.

---

## 📋 Pre-Launch Checklist

### Phase 1: Repository Setup
- [ ] Repository created on GitHub
- [ ] All files committed to main branch
- [ ] Repository is public (or accessible to students)
- [ ] Repository description added
- [ ] Topics/tags added (e.g., "reinforcement-learning", "competition", "education")

### Phase 2: Reference Prey (Public)
- [ ] Reference prey agent available in `reference_agents_source/prey_agent.py`
- [ ] Reference prey model available in `reference_agents_source/prey_final_model.zip`
- [ ] Reference prey tested locally
- [ ] Students implement predator only

### Phase 3: GitHub Actions Setup
- [ ] Workflow file created (`.github/workflows/evaluate.yml`)
- [ ] Workflow updated to use GitHub Secrets for private agents
- [ ] Deploy workflow created (`.github/workflows/deploy-pages.yml`)
- [ ] Workflow permissions set to "Read and write"
- [ ] "Allow GitHub Actions to create and approve pull requests" enabled
- [ ] GitHub Actions minutes usage checked (2000/month free)

### Phase 4: GitHub Pages
- [ ] GitHub Pages enabled in repository settings
- [ ] Source set to "GitHub Actions"
- [ ] Empty leaderboard generated and committed
- [ ] Leaderboard URL accessible
- [ ] Leaderboard displays correctly

### Phase 5: Testing
- [ ] Local dependencies installed (`pip install -r requirements.txt`)
- [ ] Template agent tested locally (`python test_agent.py template/agent.py`)
- [ ] Evaluation script tested locally
- [ ] Test submission created
- [ ] Test PR opened
- [ ] Test PR triggers workflow
- [ ] Evaluation runs successfully
- [ ] Results commented on test PR
- [ ] Leaderboard updates
- [ ] GitHub Pages deploys
- [ ] Test PR closed

### Phase 6: Documentation
- [ ] README.md reviewed for accuracy
- [ ] Repository owner/URLs updated in documentation
- [ ] Leaderboard URL updated in all docs
- [ ] Submission deadline specified
- [ ] Rules and guidelines clear
- [ ] Contact information provided
- [ ] SETUP.md reviewed
- [ ] QUICKREF.md reviewed

### Phase 7: Student Communication
- [ ] Announcement prepared
- [ ] Repository URL ready to share
- [ ] Leaderboard URL ready to share
- [ ] Submission deadline communicated
- [ ] Office hours scheduled (if needed)
- [ ] Communication channel set up (email, Slack, etc.)
- [ ] FAQ prepared

---

## 🧪 Testing Protocol

### Test 1: Local Testing
```bash
# Install dependencies
pip install -r requirements.txt

# Test template
python test_agent.py template/agent.py

# Should output: ✅ All tests passed!
```
**Status:** [ ] Pass [ ] Fail

### Test 2: Local Evaluation
```bash
# Create test submission
mkdir -p submissions/test_student
cp template/agent.py submissions/test_student/

# Run evaluation (student predator vs public prey)
python evaluate.py \
  --submission-dir submissions/test_student \
  --reference-agents-dir reference_agents \
  --output results/test.json \
  --episodes 10

# Check results
cat results/test.json
```
**Status:** [ ] Pass [ ] Fail

### Test 3: Leaderboard Generation
```bash
# Generate leaderboard
python generate_leaderboard.py

# Check output
open docs/index.html  # or your browser
```
**Status:** [ ] Pass [ ] Fail

### Test 4: Full CI/CD Pipeline
```bash
# Create test branch
git checkout -b test-submission

# Add test submission
git add submissions/test_student/
git commit -m "Test submission"
git push origin test-submission

# Create PR on GitHub
# Wait for workflow to complete
# Verify:
# - Workflow runs successfully
# - Results commented on PR
# - Leaderboard updates
# - No errors in logs
```
**Status:** [ ] Pass [ ] Fail

---

## 🔒 Security Verification

### Reference Agent Handling
- [ ] Reference prey stored in repo under `reference_agents_source/`
- [ ] Workflow copies reference prey into `reference_agents/`
- [ ] No secrets required for agents or models

### Repository Security
- [ ] No API keys or tokens in code
- [ ] No sensitive data in commits
- [ ] `.gitignore` configured properly
- [ ] Branch protection rules set (optional)
- [ ] Workflow permissions minimal

### Evaluation Security
<!-- Seeding removed from evaluation pipeline -->
- [ ] Students cannot access private agent code
- [ ] Evaluation runs in isolated environment
- [ ] File size limits enforced
- [ ] Only allowed file types accepted

---

## 📊 Performance Verification

### GitHub Actions
- [ ] Evaluation completes in <10 minutes
- [ ] No timeout errors
- [ ] Memory usage acceptable
- [ ] Workflow logs clear and helpful

### Leaderboard
- [ ] Loads quickly
- [ ] Displays correctly on mobile
- [ ] Updates within 5 minutes of commit
- [ ] Shows all required information
- [ ] Sorting works correctly

### Student Experience
- [ ] Template is clear and usable
- [ ] Test script provides helpful feedback
- [ ] PR template is helpful
- [ ] Error messages are clear
- [ ] Documentation is comprehensive

---

## Launch Day Checklist

### Morning Of
- [ ] All systems operational
- [ ] GitHub Actions running
- [ ] Leaderboard accessible
- [ ] Test submission successful
- [ ] Monitor dashboard ready

### Announcement
- [ ] Email sent to students
- [ ] Announcement posted (LMS, Slack, etc.)
- [ ] Office hours announced
- [ ] Deadline clearly stated
- [ ] Resources shared

### During Competition
- [ ] Monitor GitHub Actions for errors
- [ ] Check for stuck workflows
- [ ] Respond to student questions
- [ ] Watch for suspicious submissions
- [ ] Track GitHub Actions minutes usage

### First Submission
- [ ] Workflow runs successfully
- [ ] Results appear on PR
- [ ] Leaderboard updates
- [ ] Student receives feedback
- [ ] No errors reported

---

## 🐛 Troubleshooting Quick Reference

### Workflow Fails
1. Check Actions tab for logs
2. Verify secrets are set correctly
3. Test locally
4. Check Python version compatibility

### Leaderboard Not Updating
1. Verify commit was pushed
2. Check deploy workflow logs
3. Verify GitHub Pages enabled
4. Wait 2-5 minutes for deploy

### Private Agents Not Found
1. Verify secrets are set
2. Check secret names match workflow
3. Verify base64 encoding
4. Test decoding locally

### Evaluation Errors
1. Check student agent syntax
2. Verify agent follows template
3. Check for missing dependencies
4. Review error logs in PR

---

## 📈 Success Metrics

Track these metrics during the competition:

### Participation
- [ ] Number of students who forked
- [ ] Number of submissions received
- [ ] Number of students on leaderboard
- [ ] Submission frequency

### Technical
- [ ] Workflow success rate (target: >95%)
- [ ] Average evaluation time (target: <10 min)
- [ ] GitHub Actions minutes used
- [ ] Leaderboard uptime (target: 99%+)

### Educational
- [ ] Student engagement
- [ ] Questions asked
- [ ] Improvement over time
- [ ] Final scores distribution

---

## 🎉 Post-Competition

### Wrap-Up
- [ ] Final leaderboard screenshot saved
- [ ] Results exported
- [ ] Winners announced
- [ ] Feedback survey sent
- [ ] Lessons learned documented

### Archiving
- [ ] All results backed up
- [ ] Repository archived (optional)
- [ ] Documentation updated with lessons learned
- [ ] Student feedback collected

### Recognition
- [ ] Top performers recognized
- [ ] Most improved highlighted
- [ ] Creative approaches noted
- [ ] Feedback shared with class

---

## 📞 Emergency Contacts

**GitHub Issues**: If urgent, open an issue in the repository

**Instructor Contact**: [Your contact information]

**GitHub Status**: https://www.githubstatus.com/

**PettingZoo Issues**: https://github.com/Farama-Foundation/PettingZoo/issues

---

## Final Sign-Off

Before announcing to students:

**Instructor Name**: ________________________

**Date**: ________________________

**All Tests Passed**: [ ] Yes [ ] No

**Ready to Launch**: [ ] Yes [ ] No

**Backup Plan Ready**: [ ] Yes [ ] No

---

## Launch Command

When everything is checked and ready:

```bash
# Final commit
git add -A
git commit -m "Ready for launch"
git push origin main

# Announce to students
echo "Competition is LIVE!"
```

**Good luck with your RL competition!**

---

## Notes

Use this space for additional notes or customizations:

```

[Your notes here]

```
