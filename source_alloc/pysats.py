# Libs
import os
print(os.environ.get('JAVA_HOME'))

# %%
class PySats:
    __instance = None

    @staticmethod
    def getInstance():
        """ Static access method. """
        if PySats.__instance is None:
            PySats()
        return PySats.__instance

    def __init__(self):
        """ Virtually private constructor. """
        if PySats.__instance is not None:
            raise Exception("This class is a singleton!")
        else:
            import jnius_config
            jnius_config.set_classpath('.', os.path.join('lib', '*'))
            PySats.__instance = self

    def create_lsvm(self, seed=None, number_of_national_bidders=1, number_of_regional_bidders=5):
        from source.lsvm import _Lsvm
        return _Lsvm(seed, number_of_national_bidders, number_of_regional_bidders)

    def create_gsvm(self, seed=None, number_of_national_bidders=1, number_of_regional_bidders=6):
        from source.gsvm import _Gsvm
        return _Gsvm(seed, number_of_national_bidders, number_of_regional_bidders)

    def create_mrvm(self, seed=None, number_of_national_bidders=3, number_of_regional_bidders=4, number_of_local_bidders=3):
        from source.mrvm import _Mrvm
        return _Mrvm(seed, number_of_national_bidders, number_of_regional_bidders, number_of_local_bidders)

    def create_srvm(self, seed=None, number_of_small_bidders=2, number_of_high_frequency_bidders=1, number_of_secondary_bidders=2, number_of_primary_bidders=2):
        from source.srvm import _Srvm
        return _Srvm(seed, number_of_small_bidders, number_of_high_frequency_bidders,
                     number_of_secondary_bidders, number_of_primary_bidders)