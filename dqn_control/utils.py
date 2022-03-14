

# prints an execution report of a single controlled execution.
# The interactions between the system and the environment are written one after the other,
# and eventually the fail rate of the execution appears.
def print_experiment(output_file, system_execution, environment_execution, fail_rate):
    with open(output_file, 'w') as f:
        titles = ['Step', 'System', 'Environment']
        print('{:^6}|   {:^60}|   {:^60}'.format(*titles), file=f)

        for i, item in enumerate(zip(system_execution, environment_execution)):
            print('{:<6}|   {:<60}|   {:<60}'.format(i + 1, str(item[0]), str(item[1])), file=f)

        print('', file=f)
        print("Fail rate is: {0}%".format(fail_rate * 100), file=f)

def print_dual_system_experiment(output_file, system1_execution, system2_execution, fail_rate):
    with open(output_file, 'w') as f:
        titles = ['Step', 'System 1', 'System 2']
        print('{:^6}|   {:^60}|   {:^60}'.format(*titles), file=f)

        for i, item in enumerate(zip(system1_execution, system2_execution)):
            print('{:<6}|   {:<60}|   {:<60}'.format(i + 1, str(item[0]), str(item[1])), file=f)

        print('', file=f)
        print("Fail rate is: {0}%".format(fail_rate * 100), file=f)
