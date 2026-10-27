use super::*;

impl<T: Element + Clone> Buffer<T> {
    // Only fresh ownership or a successful Arc uniqueness check establishes this
    // fact. Clone and keepalive retire it before an alias can escape. No mutable
    // access relies on it: make_mut still goes through safe Arc::make_mut.
    #[inline]
    pub(super) fn has_owned_capacity(&self, additional: usize) -> bool {
        match &self.data {
            Data::Owned { values, unique } => {
                unique.load(Ordering::Relaxed) && values.capacity() - values.len() >= additional
            }
            Data::Borrowed { .. } => false,
        }
    }

    /// Prepare owned capacity without changing values or remembered input facts.
    #[inline]
    pub(crate) fn try_reserve(&mut self, additional: usize) -> crate::Result<()> {
        if self.has_owned_capacity(additional) { return Ok(()); }
        self.reserve_slow(additional)
    }

    #[cold]
    #[inline(never)]
    fn reserve_slow(&mut self, additional: usize) -> crate::Result<()> {
        if let Data::Owned { values, unique } = &mut self.data {
            if let Some(values) = Arc::get_mut(values) {
                values.try_reserve(additional).map_err(crate::column::capacity::allocation_error)?;
                unique.store(true, Ordering::Relaxed);
                return Ok(());
            }
        }
        let len = self.len();
        let needed = len.checked_add(additional).ok_or_else(crate::column::capacity::capacity_error)?;
        // A COW copy must keep the spare that clone-then-append would retain.
        let mut values = Vec::new();
        values.try_reserve(crate::capacity::growth_capacity(len, needed))
            .map_err(crate::column::capacity::allocation_error)?;
        values.extend_from_slice(self.as_slice());
        self.data = Data::owned(values);
        Ok(())
    }
}
