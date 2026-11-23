"""Print the number of ranges still to do for a release and a set of phases (used to chain workflow runs)."""
import os, sys
import dr4engine as E
rel, phases, done_dir = sys.argv[1], set(sys.argv[2].split(",")), sys.argv[3]
r = E.discover(rel, "cache")
done = set(os.listdir(done_dir)) if os.path.isdir(done_dir) else set()
print(sum(1 for k in r if E.phase_of(k) in phases and k not in done))
