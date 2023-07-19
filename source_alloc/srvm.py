

# Libs
from jnius import JavaClass, MetaJavaClass, JavaMethod, cast, autoclass



# %%
SizeBasedUniqueRandomXOR = autoclass(
    'org.spectrumauctions.sats.core.bidlang.xor.SizeBasedUniqueRandomXOR')
JavaUtilRNGSupplier = autoclass(
    'org.spectrumauctions.sats.core.util.random.JavaUtilRNGSupplier')
Bundle = autoclass(
    'org.spectrumauctions.sats.core.model.Bundle')

SRVM_MIP = autoclass(
    'org.spectrumauctions.sats.opt.model.srvm.SRVM_MIP')


class _Srvm(JavaClass, metaclass=MetaJavaClass):
    __javaclass__ = 'org/spectrumauctions/sats/core/model/srvm/SingleRegionModel'

    # TODO: I don't find a way to have the more direct accessors of the DefaultModel class. So for now, I'm mirroring the accessors
    #createNewPopulation = JavaMultipleMethod([
    #    '()Ljava/util/List;',
    #    '(J)Ljava/util/List;'])
    setNumberOfSmallBidders = JavaMethod('(I)V')
    setNumberOfHighFrequencyBidders = JavaMethod('(I)V')
    setNumberOfSecondaryBidders = JavaMethod('(I)V')
    setNumberOfPrimaryBidders = JavaMethod('(I)V')
    createWorld = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Lorg/spectrumauctions/sats/core/model/srvm/SRVMWorld;')
    createPopulation = JavaMethod(
        '(Lorg/spectrumauctions/sats/core/model/World;Lorg/spectrumauctions/sats/core/util/random/RNGSupplier;)Ljava/util/List;')

    population = {}
    goods = {}
    efficient_allocation = None

    def __init__(self, seed, number_of_small_bidders, number_of_high_frequency_bidders, number_of_secondary_bidders, number_of_primary_bidders):
        super().__init__()
        if seed:
            rng = JavaUtilRNGSupplier(seed)
        else:
            rng = JavaUtilRNGSupplier()

        self.number_of_small_bidders = number_of_small_bidders
        self.number_of_high_frequency_bidders = number_of_high_frequency_bidders
        self.number_of_secondary_bidders = number_of_secondary_bidders
        self.number_of_primary_bidders = number_of_primary_bidders

        world = self.createWorld(rng)
        self._bidder_list = self.createPopulation(world, rng)

        # Store bidders
        bidderator = self._bidder_list.iterator()
        while bidderator.hasNext():
            bidder = bidderator.next()
            self.population[bidder.getId()] = bidder

        # Store goods
        goods_iterator = self._bidder_list.iterator().next().getWorld().getLicenses().iterator()
        while goods_iterator.hasNext():
            good = goods_iterator.next()
            self.goods[good.getId()] = good

        self.goods = list(map(lambda _id: self.goods[_id], sorted(self.goods.keys())))

    def get_bidder_ids(self):
        return self.population.keys()

    def get_good_ids(self):
        return dict.fromkeys(list(range(29))).keys()

    def calculate_value(self, bidder_id, goods_vector):
        assert len(goods_vector) == len(self.goods)
        bidder = self.population[bidder_id]
        bundle = Bundle()
        for i in range(len(goods_vector)):
            if goods_vector[i] == 1:
                bundle.add(self.goods[i])
        return bidder.calculateValue(bundle).doubleValue()

    def get_random_bids(self, bidder_id, number_of_bids, seed=None, mean_bundle_size=49, standard_deviation_bundle_size=24.5):
        bidder = self.population[bidder_id]
        if seed:
            rng = JavaUtilRNGSupplier(seed)
        else:
            rng = JavaUtilRNGSupplier()
        valueFunction = cast('org.spectrumauctions.sats.core.bidlang.xor.SizeBasedUniqueRandomXOR',
                             bidder.getValueFunction(SizeBasedUniqueRandomXOR, rng))
        valueFunction.setDistribution(
            mean_bundle_size, standard_deviation_bundle_size)
        valueFunction.setIterations(number_of_bids)
        xorBidIterator = valueFunction.iterator()
        bids = []
        while (xorBidIterator.hasNext()):
            xorBid = xorBidIterator.next()
            bid = []
            for i in range(len(self.goods)):
                if (xorBid.getLicenses().contains(self.goods[i])):
                    bid.append(1)
                else:
                    bid.append(0)
            bid.append(xorBid.value)
            bids.append(bid)
        return bids

    def get_efficient_allocation(self):
        if self.efficient_allocation:
            return self.efficient_allocation, sum([self.efficient_allocation[bidder_id]['value'] for bidder_id in self.efficient_allocation.keys()])

        mip = SRVM_MIP(self._bidder_list)
        mip.setDisplayOutput(True)

        generic_allocation = cast(
            'org.spectrumauctions.sats.opt.domain.GenericAllocation', mip.calculateAllocation())

        self.efficient_allocation = {}

        for bidder_id, bidder in self.population.items():
            self.efficient_allocation[bidder_id] = {}
            self.efficient_allocation[bidder_id]['good_ids'] = []
            if generic_allocation.getWinners().contains(bidder):
                bidder_allocation = generic_allocation.getAllocation(bidder)
                good_iterator = bidder_allocation.iterator()
                while good_iterator.hasNext():
                    self.efficient_allocation[bidder_id]['good_ids'].append(good_iterator.next().getId())

            self.efficient_allocation[bidder_id]['value'] = generic_allocation.getTradeValue(
                bidder).doubleValue()

        return self.efficient_allocation, generic_allocation.totalValue.doubleValue()
